"""
Dell-AITC Keyword Classifier Service
Handles keyword-based classification of use cases.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from ...models.ai_category import AICategory
from ...config import get_settings
from ..database.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

class KeywordClassifier:
    """Keyword-based classifier for AI technology categories"""
    
    def __init__(self):
        """Initialize classifier"""
        self.categories = {}
        self.db_service = Neo4jService()
        self.logger = logging.getLogger(__name__)
        
    async def get_categories(self) -> List[AICategory]:
        """Get AI categories from Neo4j
        
        Returns:
            List of AICategory objects
        """
        query = """
        MATCH (c:AICategory)
        RETURN {
            name: c.name,
            keywords: c.keywords,
            description: c.description,
            maturity_level: c.maturity_level,
            capabilities: c.capabilities,
            business_language: c.business_language
        } as category
        """
        
        results = await self.db_service.run_query(query)
        categories = []
        
        for record in results:
            cat_data = record['category']
            categories.append(AICategory(
                name=cat_data['name'],
                keywords=cat_data.get('keywords', []),
                description=cat_data.get('description', ''),
                maturity_level=cat_data.get('maturity_level', ''),
                capabilities=cat_data.get('capabilities', []),
                business_language=cat_data.get('business_language', [])
            ))
            
        return categories
        
    async def initialize(self):
        """Initialize classifier components"""
        self.logger.info("Initializing keyword classifier")
        
        # Load categories
        categories = await self.get_categories()
        self.categories = {cat.name: cat for cat in categories}
        self.logger.info(f"Loaded {len(self.categories)} categories")
            
    async def cleanup(self):
        """Clean up resources"""
        if self.db_service:
            await self.db_service.cleanup()
            
    async def get_unclassified_use_cases(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get use cases that haven't been classified yet
        
        Args:
            limit: Optional maximum number of cases to return
            
        Returns:
            List of unclassified use case dictionaries
        """
        query = """
        MATCH (u:UseCase)
        WHERE NOT EXISTS((u)-[:CLASSIFIED_AS]->(:AICategory))
        RETURN {
            id: u.id,
            name: u.name,
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            status: u.status,
            dev_stage: u.dev_stage,
            dev_method: u.dev_method
        } as use_case
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        results = await self.db_service.run_query(query)
        return [record['use_case'] for record in results]
        
    def _calculate_term_matches(self, text: str, category: AICategory) -> Tuple[List[str], List[float], Dict[str, List[str]]]:
        """Calculate term matches between text and category keywords
        
        Args:
            text: Text to analyze
            category: Category to match against
            
        Returns:
            Tuple of (matched terms, scores, field matches)
        """
        if not text:
            return [], [], {}
            
        text = text.lower()
        matched_terms = []
        scores = []
        field_matches = {}
        
        # Check each keyword
        for keyword in category.keywords:
            keyword_text = keyword.lower()
            
            # Exact match
            if keyword_text in text:
                matched_terms.append(keyword)
                scores.append(1.0)
                continue
                
            # Word match
            if any(keyword_text in word or word in keyword_text 
                  for word in text.split()):
                matched_terms.append(keyword)
                scores.append(0.8)
                continue
                
            # Partial match
            if any(part in text for part in keyword_text.split()):
                matched_terms.append(keyword)
                scores.append(0.5)
                
        return matched_terms, scores, field_matches
        
    def classify_text(self, text: str, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Classify text against all categories
        
        Args:
            text: Text to classify
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matches above threshold
        """
        matches = []
        
        for category in self.categories.values():
            # Get term matches
            matched_terms, scores, field_matches = self._calculate_term_matches(text, category)
            
            if not matched_terms:
                continue
                
            # Calculate confidence
            confidence = sum(scores) / len(category.keywords) if category.keywords else 0.0
            
            if confidence >= min_confidence:
                matches.append({
                    "category": category.name,
                    "confidence": confidence,
                    "matched_terms": matched_terms,
                    "field_matches": field_matches
                })
                
        # Sort by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches
        
    def analyze_unmatched(self, text: str) -> Dict[str, Any]:
        """Analyze text that didn't match any categories
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        # Get all matches regardless of confidence
        all_matches = self.classify_text(text, min_confidence=0.0)
        
        # Get top 3 closest matches
        closest_matches = all_matches[:3]
        
        return {
            "closest_matches": closest_matches,
            "extracted_terms": self._extract_technical_terms(text),
            "suggested_keywords": self._suggest_keywords(text)
        }
        
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of technical terms
        """
        # Simple extraction based on common technical indicators
        technical_indicators = {
            'ai', 'ml', 'api', 'data', 'model', 'algorithm', 'system',
            'neural', 'cloud', 'platform', 'framework', 'analytics'
        }
        
        words = text.lower().split()
        technical_terms = []
        
        for i, word in enumerate(words):
            if word in technical_indicators:
                # Get surrounding context
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                term = ' '.join(words[start:end])
                technical_terms.append(term)
                
        return list(set(technical_terms))
        
    def _suggest_keywords(self, text: str) -> List[str]:
        """Suggest keywords based on text content
        
        Args:
            text: Text to analyze
            
        Returns:
            List of suggested keywords
        """
        # Get technical terms
        technical_terms = self._extract_technical_terms(text)
        
        # Get existing keywords from all categories
        existing_keywords = set()
        for category in self.categories.values():
            existing_keywords.update(category.keywords)
            
        # Filter out terms that are already keywords
        suggestions = []
        for term in technical_terms:
            if not any(term in keyword.lower() for keyword in existing_keywords):
                suggestions.append(term)
                
        return suggestions 