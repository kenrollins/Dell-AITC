"""
Dell-AITC Classifier Service (v2.2)
Handles classification of use cases against AI technology categories.
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from ..models.analysis import AnalysisMethod
from ..models.ai_category import get_categories, AICategory
from ..models.match_type import MatchType, MatchResult
from ..utils.text_processor import TextProcessor
from ..services.llm_analyzer import LLMAnalyzer
from neo4j import GraphDatabase
from ..config import Settings, get_settings
from neo4j import AsyncGraphDatabase
import httpx
from openai import AsyncOpenAI, OpenAI
import asyncio

logger = logging.getLogger(__name__)

class Classifier:
    def __init__(self, dry_run: bool = False):
        """Initialize classifier with configuration."""
        self.settings = get_settings()
        self.dry_run = dry_run
        self.driver = None
        self.text_processor = None
        self.llm_analyzer = None
        self.categories = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self, api_timeout: float = 60.0):
        """Initialize connections and load dependencies."""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
            
        if not self.text_processor:
            self.text_processor = TextProcessor()
            
        if not self.llm_analyzer:
            self.llm_analyzer = LLMAnalyzer()
            await self.llm_analyzer.initialize()
            
        if not self.categories:
            self.categories = await get_categories()
            
        # Initialize OpenAI client with configurable timeout
        self.openai_client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=api_timeout,
            max_retries=2
        )
            
        logging.info(f"Classifier initialized with {len(self.categories)} categories")

    def _calculate_keyword_score(self, use_case_text: str, category: AICategory) -> float:
        """Calculate keyword-based score."""
        try:
            # Get category keywords
            keywords = category.keywords
            if not keywords:
                return 0.0
            
            # Convert to lowercase for case-insensitive matching
            use_case_text = use_case_text.lower()
            
            # Track matches and partial matches
            exact_matches = 0
            partial_matches = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Check for exact match
                if keyword_lower in use_case_text:
                    exact_matches += 1
                    continue
                
                # Handle compound terms
                keyword_parts = keyword_lower.split()
                if len(keyword_parts) > 1:
                    # Check if all parts appear in text in any order
                    if all(part in use_case_text for part in keyword_parts):
                        exact_matches += 0.8
                        continue
                    
                    # Check if most parts appear
                    matched_parts = sum(1 for part in keyword_parts if part in use_case_text)
                    if matched_parts / len(keyword_parts) >= 0.6:
                        partial_matches += 0.6
                        continue
                
                # Partial matching for single terms
                if len(keyword_lower) > 4:
                    if any(keyword_lower[i:i+len(keyword_lower)] in use_case_text 
                          for i in range(len(keyword_lower)) 
                          if len(keyword_lower[i:i+len(keyword_lower)]) >= 4):
                        partial_matches += 0.4
            
            # Calculate final score with reduced penalty for keyword count
            total_score = exact_matches + (partial_matches * 0.7)
            max_possible = max(len(keywords) * 0.4, 1.0)  # Reduced denominator
            
            return min(1.0, total_score / max_possible)
            
        except Exception as e:
            logging.error(f"Error calculating keyword score: {str(e)}")
            return 0.0

    def _evaluate_matches(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, List[Dict[str, Any]]]:
        """Evaluate and categorize matches with adjusted thresholds and weights."""
        matches = {
            'primary': [],
            'supporting': [],
            'related': []
        }
        
        for category_name, category_scores in scores.items():
            # Get individual scores
            keyword_score = category_scores.get('keyword_score', 0.0)
            semantic_score = category_scores.get('semantic_score', 0.0)
            llm_score = category_scores.get('llm_score', 0.0)
            
            # Calculate method agreement (lowered threshold)
            methods_above_threshold = sum([
                1 for score in [keyword_score, semantic_score, llm_score]
                if score >= 0.25  # Significantly lowered from 0.3
            ])
            
            # Calculate weighted scores (adjusted weights)
            weighted_scores = {
                'keyword': keyword_score * 0.45,    # Increased from 0.4
                'semantic': semantic_score * 0.35,   # Same
                'llm': llm_score * 0.20             # Same
            }
            
            # Calculate agreement boost (increased maximum)
            agreement_boost = min(0.3, methods_above_threshold * 0.1)  # Increased from 0.25
            
            # Calculate blended scores with increased keyword influence
            base_score = max(
                keyword_score * 0.8 + semantic_score * 0.2,   # Increased keyword weight
                semantic_score * 0.6 + llm_score * 0.4,       # Same
                keyword_score * 0.7 + llm_score * 0.3,        # Increased keyword weight
                sum(weighted_scores.values())                  # All methods weighted
            )
            
            # Apply agreement boost
            final_score = min(1.0, base_score * (1 + agreement_boost))
            
            # Add individual method bonuses (lowered thresholds)
            if keyword_score >= 0.5:   # Lowered from 0.6
                final_score = min(1.0, final_score + 0.15)  # Increased bonus
            if semantic_score >= 0.5:  # Lowered from 0.6
                final_score = min(1.0, final_score + 0.1)   # Increased bonus
            if llm_score >= 0.6:      # Lowered from 0.65
                final_score = min(1.0, final_score + 0.08)  # Increased bonus
            
            match_details = {
                'category_name': category_name,
                'confidence': final_score,
                'scores': {
                    'keyword': keyword_score,
                    'semantic': semantic_score,
                    'llm': llm_score,
                    'base_score': base_score,
                    'agreement_boost': agreement_boost
                },
                'method_agreement': methods_above_threshold
            }
            
            # Categorize match (lowered thresholds)
            if final_score >= 0.55 or (final_score >= 0.5 and methods_above_threshold >= 2):  # Lowered from 0.6/0.55
                matches['primary'].append(match_details)
            elif final_score >= 0.35 or (final_score >= 0.3 and methods_above_threshold >= 2):  # Same
                matches['supporting'].append(match_details)
            elif final_score >= 0.25 or methods_above_threshold >= 1:  # Lowered from 0.3
                matches['related'].append(match_details)
            
            # Sort matches by confidence
            for match_type in matches:
                matches[match_type].sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches

    async def classify_use_case(self, use_case: Dict[str, Any], method: AnalysisMethod = AnalysisMethod.ALL, save_to_db: bool = True) -> MatchResult:
        """
        Classify a use case with support for multiple matches per type.
        Returns a MatchResult containing all matches found.
        """
        try:
            # Initialize results structure
            results = MatchResult(
                use_case_id=use_case['id'],
                primary_matches=[],
                supporting_matches=[],
                related_matches=[],
                match_method='ENSEMBLE',
                confidence=0.0,
                field_match_scores={},
                matched_terms={},
                llm_analysis={}
            )
            
            # Collect scores for all categories
            category_scores = {}
            for category_name, category in self.categories.items():
                scores = {}
                
                # Keyword analysis
                keyword_results = self._calculate_keyword_score(use_case['description'], category)
                scores['keyword_score'] = keyword_results
                results.matched_terms[category_name] = keyword_results
                
                # Semantic analysis
                if method in [AnalysisMethod.SEMANTIC, AnalysisMethod.ALL]:
                    semantic_results = self._perform_semantic_analysis(use_case)
                    scores['semantic_score'] = semantic_results.get(category_name, {}).get('score', 0.0)
                    
                # LLM analysis if needed
                if method in [AnalysisMethod.LLM, AnalysisMethod.ALL]:
                    # Lower thresholds for LLM analysis to get more potential matches
                    if scores['keyword_score'] >= 0.3 or scores['semantic_score'] >= 0.35:
                        llm_results = await self._perform_llm_analysis(use_case)
                        scores['llm_score'] = llm_results.get('llm_score', 0.0)
                        if llm_results.get('llm_explanation'):
                            results.llm_analysis[category_name] = llm_results['llm_explanation']
                
                category_scores[category_name] = scores
            
            # Evaluate and categorize matches
            matches = self._evaluate_matches(category_scores)
            results.primary_matches = matches['primary']
            results.supporting_matches = matches['supporting']
            results.related_matches = matches['related']
            
            # Set overall confidence from best match
            best_match = results.get_best_match()
            if best_match:
                results.confidence = best_match['confidence']
            
            # Save to Neo4j if requested
            if save_to_db and not self.dry_run:
                await self._save_to_neo4j(results)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return MatchResult(
                use_case_id=use_case['id'],
                primary_matches=[],
                supporting_matches=[],
                related_matches=[],
                match_method='ERROR',
                confidence=0.0,
                field_match_scores={},
                matched_terms={},
                error=str(e)
            )

    async def close(self):
        """Cleanup connections."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            
        if self.llm_analyzer:
            await self.llm_analyzer.cleanup()
            self.llm_analyzer = None
            
    def __del__(self):
        """Ensure connections are closed."""
        if self.driver:
            try:
                asyncio.create_task(self.close())
                logging.debug("Classifier cleanup initiated")
            except Exception as e:
                logging.error(f"Error during classifier cleanup: {str(e)}")

    async def get_use_case_by_id(self, use_case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a use case by ID from Neo4j.
        
        Args:
            use_case_id: The ID of the use case to fetch
            
        Returns:
            Dictionary containing use case data or None if not found
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (u:UseCase {id: $use_case_id})
                RETURN {
                    id: u.id,
                    name: u.name,
                    description: u.description,
                    purpose_benefits: u.purpose_benefits,
                    outputs: u.outputs,
                    topic_area: u.topic_area,
                    stage: u.stage,
                    impact_type: u.impact_type,
                    dev_method: u.dev_method,
                    system_name: u.system_name
                } as use_case
                """
                result = await session.run(query, use_case_id=use_case_id)
                record = await result.single()
                
                if not record:
                    self.logger.error(f"Use case {use_case_id} not found")
                    return None
                    
                return record["use_case"]
                
        except Exception as e:
            self.logger.error(f"Error fetching use case {use_case_id}: {str(e)}")
            return None

    def classify_by_use_case_number(self, use_case_number: str, method: AnalysisMethod = AnalysisMethod.ALL, save_to_db: bool = True) -> Optional[Dict[str, Any]]:
        """
        Classify a specific use case by its number/ID
        
        Args:
            use_case_number: The use case ID to classify
            method: Analysis method to use
            save_to_db: Whether to save results to Neo4j
        """
        with self.driver.session() as session:
            # Fetch use case data
            query = """
            MATCH (u:UseCase {id: $use_case_id})
            RETURN {
                id: u.id,
                name: u.name,
                description: u.description,
                purpose_benefits: u.purpose_benefits,
                outputs: u.outputs
            } as use_case
            """
            result = session.run(query, use_case_id=use_case_number)
            record = result.single()
            
            if not record:
                self.logger.error(f"Use case {use_case_number} not found")
                return None
                
            use_case = record['use_case']
            # In dry run mode, we classify but don't save
            return self.classify_use_case(use_case, method, save_to_db=not self.dry_run)

    async def _save_to_neo4j(self, results: MatchResult) -> None:
        """Save classification results with multiple matches to Neo4j"""
        if self.dry_run:
            return
            
        async with self.driver.session() as session:
            # Save primary matches
            for idx, match in enumerate(results.primary_matches):
                await self._create_classification_node(
                    session,
                    results.use_case_id,
                    match['category_name'],
                    'PRIMARY',
                    idx + 1,  # rank
                    match['confidence'],
                    match['scores'],
                    results.matched_terms.get(match['category_name'], [])
                )
                
            # Save supporting matches
            for idx, match in enumerate(results.supporting_matches):
                await self._create_classification_node(
                    session,
                    results.use_case_id,
                    match['category_name'],
                    'SUPPORTING',
                    idx + 1,
                    match['confidence'],
                    match['scores'],
                    results.matched_terms.get(match['category_name'], [])
                )
                
            # Save related matches
            for idx, match in enumerate(results.related_matches):
                await self._create_classification_node(
                    session,
                    results.use_case_id,
                    match['category_name'],
                    'RELATED',
                    idx + 1,
                    match['confidence'],
                    match['scores'],
                    results.matched_terms.get(match['category_name'], [])
                )

    async def _create_classification_node(
        self,
        session,
        use_case_id: str,
        category_name: str,
        match_type: str,
        rank: int,
        confidence: float,
        scores: Dict[str, float],
        matched_terms: List[str]
    ) -> None:
        """Create a classification node with relationships"""
        query = """
        MATCH (u:UseCase {id: $use_case_id})
        MATCH (c:AICategory {name: $category_name})
        
        CREATE (cl:AIClassification {
            id: apoc.create.uuid(),
            match_type: $match_type,
            match_rank: $rank,
            confidence: $confidence,
            analysis_method: 'ENSEMBLE',
            analysis_version: 'v2.2',
            keyword_score: $keyword_score,
            semantic_score: $semantic_score,
            llm_score: $llm_score,
            matched_terms: $matched_terms,
            classified_at: datetime(),
            classified_by: 'system',
            last_updated: datetime()
        })
        
        CREATE (c)-[:CLASSIFIES]->(cl)
        CREATE (u)-[:CLASSIFIED_AS]->(cl)
        
        RETURN cl
        """
        
        await session.run(
            query,
            use_case_id=use_case_id,
            category_name=category_name,
            match_type=match_type,
            rank=rank,
            confidence=confidence,
            keyword_score=scores.get('keyword', 0.0),
            semantic_score=scores.get('semantic', 0.0),
            llm_score=scores.get('llm', 0.0),
            matched_terms=matched_terms
        )

    def _perform_keyword_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """Perform keyword-based analysis with improved confidence scoring"""
        # Combine relevant text fields for analysis, safely handling None values
        text_fields = {
            'name': str(use_case.get('name', '') or '').lower(),
            'description': str(use_case.get('description', '') or '').lower(),
            'purpose_benefits': str(use_case.get('purpose_benefits', '') or '').lower(),
            'outputs': str(use_case.get('outputs', '') or '').lower()
        }
        
        # Field weights (adjusted for better balance)
        field_weights = {
            'name': 3.0,        # High weight for name matches
            'description': 1.0,  # Base weight for description
            'purpose_benefits': 2.0,  # Higher weight for purpose
            'outputs': 1.5      # Medium weight for outputs
        }
        
        best_match: Tuple[str, float, List[str], Dict[str, Any]] = ('', 0.0, [], {})
        
        # Check each category
        for category_name, category in self.categories.items():
            matched_keywords = []
            field_matches = {field: [] for field in text_fields.keys()}
            term_weights = {}
            
            # Check each field
            for field_name, field_text in text_fields.items():
                # Technical keywords (highest weight)
                for keyword in category.keywords:
                    if not keyword:  # Skip empty keywords
                        continue
                    keyword_lower = str(keyword).lower()
                    if keyword_lower in field_text:
                        matched_keywords.append(keyword)
                        field_matches[field_name].append(keyword)
                        # Boost weight for exact matches in name or purpose
                        if field_name in ['name', 'purpose_benefits'] and keyword_lower == field_text:
                            term_weights[keyword] = 2.0 * field_weights[field_name]
                        else:
                            term_weights[keyword] = 1.5 * field_weights[field_name]
                
                # Capability terms (medium weight)
                for capability in category.capabilities:
                    if not capability:  # Skip empty capabilities
                        continue
                    capability_lower = str(capability).lower()
                    if capability_lower in field_text:
                        matched_keywords.append(capability)
                        field_matches[field_name].append(capability)
                        term_weights[capability] = 1.2 * field_weights[field_name]
                
                # Business language terms (lower weight)
                for term in category.business_language:
                    if not term:  # Skip empty terms
                        continue
                    term_lower = str(term).lower()
                    if term_lower in field_text:
                        matched_keywords.append(term)
                        field_matches[field_name].append(term)
                        term_weights[term] = 0.9 * field_weights[field_name]
            
            # Calculate score if we have matches
            if matched_keywords:
                # Remove duplicates while preserving order
                matched_keywords = list(dict.fromkeys(matched_keywords))
                
                # Base score from term weights (adjusted calculation)
                base_score = sum(term_weights[kw] for kw in matched_keywords)
                # Normalize by maximum possible score instead of keyword count
                max_possible_score = max(len(category.keywords) * 1.5 * max(field_weights.values()), 1)
                base_score = base_score / max_possible_score
                
                # Calculate match density (adjusted)
                field_densities = {
                    field: len(matches) / max(len(field_text.split()) / 4, 1)  # Reduced density requirement
                    for field, matches in field_matches.items()
                    for field_text in [text_fields[field]]
                    if matches
                }
                avg_density = sum(field_densities.values()) / len(field_densities) if field_densities else 0
                
                # Calculate diversity score (adjusted weight)
                fields_with_matches = sum(1 for matches in field_matches.values() if matches)
                field_diversity = (fields_with_matches / len(field_matches)) * 1.5  # Increased diversity impact
                
                # Combine scores with adjusted weights
                final_score = (
                    base_score * 0.6 +  # Reduced base score weight
                    avg_density * 0.2 +
                    field_diversity * 0.2  # Increased diversity weight
                )
                
                # Add score boosts
                if fields_with_matches >= 2:
                    final_score *= 1.2  # Reduced multi-field boost
                
                if len(matched_keywords) >= 3:
                    final_score *= 1.3  # Reduced keyword count boost
                
                # Boost score for exact category name matches
                if category_name.lower() in text_fields['name'].lower():
                    final_score *= 1.3  # Reduced exact match boost
                
                # Ensure score is between 0 and 1
                final_score = min(1.0, final_score)
                
                # Update best match if this score is higher
                if final_score > best_match[1]:
                    best_match = (
                        category_name,
                        final_score,
                        matched_keywords,
                        {
                            'field_matches': field_matches,
                            'term_weights': term_weights,
                            'density_score': avg_density,
                            'diversity_score': field_diversity,
                            'base_score': base_score
                        }
                    )
        
        # Return results with category name if score is high enough
        return {
            'keyword_score': best_match[1],
            'matched_keywords': best_match[2],
            'category_name': best_match[0] if best_match[1] > 0.2 else None,  # Lowered threshold
            'match_details': best_match[3] if best_match[1] > 0.2 else {},
            '_best_match_category': best_match[0]  # Store best match category regardless of threshold
        }

    def _perform_semantic_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic analysis using sentence transformers"""
        # Initialize text processor if not already done
        if not hasattr(self, 'text_processor'):
            self.text_processor = TextProcessor()
            
        # Format use case text as a dictionary
        use_case_text = {
            'description': use_case.get('description', ''),
            'purpose_benefits': use_case.get('purpose_benefits', ''),
            'outputs': use_case.get('outputs', ''),
            'systems': use_case.get('name', '')  # Using name as systems field
        }
        
        results = {}
        
        # Check each category
        for category_name, category in self.categories.items():
            # Format category text as a dictionary
            category_text = {
                'description': f"{category.name}: {' '.join(category.keywords)}",
                'purpose_benefits': ' '.join(category.business_language),
                'outputs': ' '.join(category.capabilities),
                'systems': category.name
            }
            
            score = self.text_processor.calculate_weighted_semantic_score(
                use_case_text=use_case_text,
                category_text=category_text
            )
            
            results[category_name] = {
                'score': score,
                'semantic_score': score,
                'confidence': score if score > 0.30 else 0.0
            }
        
        return results

    async def _perform_llm_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM-based analysis comparing the use case against all AI categories.
        Uses a structured approach considering all relevant use case fields and category details.
        """
        try:
            # Format use case text with all relevant fields
            use_case_text = self._format_use_case_text(use_case)
            
            # Get all active categories formatted for comparison
            categories_text = self._format_categories_for_prompt()
            
            # Prepare the analysis prompt
            messages = [
                {"role": "system", "content": """You are an expert AI technology analyst evaluating federal use cases.
Your task is to determine if the use case matches any of the available AI technology categories.

Provide your response in JSON format with the following structure:
{
    "best_match": {
        "category_name": string,       // Name of best matching category, or null if no match
        "confidence": float,           // Confidence score (0.0-1.0)
        "match_type": string,          // "PRIMARY", "SUPPORTING", "RELATED", or "NONE"
        "explanation": string          // 2-3 sentences explaining why this category matches (or why no categories match)
                                      // For matches: Focus on technical alignment and key capabilities
                                      // For no matches: Explain key technical gaps or misalignments
    },
    "field_analysis": {
        "technical_alignment": float,   // Score for technical alignment (0.0-1.0)
        "business_alignment": float,    // Score for business alignment (0.0-1.0)
        "implementation_fit": float     // Score for implementation approach fit (0.0-1.0)
    },
    "matched_terms": {
        "technical_terms": string[],    // Key technical terms that influenced the decision
        "capabilities": string[]        // Matched capabilities that influenced the decision
    }
}"""},
                {"role": "user", "content": str(f"""
EVALUATION CRITERIA:
1. Technical Alignment:
   - Does the use case implementation match the core technical capabilities of any category?
   - Are the required technical components present?
   - Do the outputs align with category-specific technical outcomes?

2. Key Capabilities:
   - What specific AI/ML capabilities are required?
   - Which category best matches these capability needs?
   - Are there any critical capability gaps?

USE CASE DETAILS:
{use_case_text}

AVAILABLE AI TECHNOLOGY CATEGORIES:
{categories_text}

Analyze this use case and determine if it matches any of the available categories.
Focus on technical alignment and required capabilities.
If no strong technical match exists, explain the key technical gaps or misalignments.""")}
            ]

            # Get LLM analysis
            response = await self.llm_analyzer._call_openai(messages)
            
            # Extract and validate the best match details
            best_match = response.get("best_match", {})
            field_scores = response.get("field_analysis", {})
            matched_terms = response.get("matched_terms", {})
            
            # Ensure field scores are floats
            field_scores = {
                "technical_alignment": float(field_scores.get("technical_alignment", 0.0)),
                "business_alignment": float(field_scores.get("business_alignment", 0.0)),
                "implementation_fit": float(field_scores.get("implementation_fit", 0.0))
            }
            
            # Process matched terms
            technical_terms = []
            capabilities = []

            # Handle technical terms
            raw_technical_terms = matched_terms.get('technical_terms', [])
            if isinstance(raw_technical_terms, (str, dict)):
                raw_technical_terms = [raw_technical_terms]
            
            for term in raw_technical_terms:
                if isinstance(term, dict):
                    term_value = term.get('name') or term.get('value') or str(term)
                    technical_terms.append(str(term_value))
                else:
                    technical_terms.append(str(term))

            # Handle capabilities
            raw_capabilities = matched_terms.get('capabilities', [])
            if isinstance(raw_capabilities, (str, dict)):
                raw_capabilities = [raw_capabilities]
            
            for cap in raw_capabilities:
                if isinstance(cap, dict):
                    cap_value = cap.get('name') or cap.get('value') or str(cap)
                    capabilities.append(str(cap_value))
                else:
                    capabilities.append(str(cap))

            # Calculate overall LLM score as weighted average of field scores
            weights = {
                "technical_alignment": 0.5,    # Increased weight for technical alignment
                "business_alignment": 0.2,     # Reduced weight for business alignment
                "implementation_fit": 0.3      # Moderate weight for implementation
            }
            
            llm_score = sum(
                field_scores.get(field, 0.0) * weight 
                for field, weight in weights.items()
            )
            
            # Apply term match boost
            term_match_boost = 0.0
            if matched_terms:
                technical_terms_count = len(technical_terms)
                capabilities_count = len(capabilities)
                
                # Add 10% boost for each technical term up to 30%
                term_match_boost = min(0.3, (technical_terms_count + capabilities_count) * 0.1)
                
            llm_score = min(1.0, llm_score * (1 + term_match_boost))
            
            return {
                'llm_score': llm_score,
                'category_name': best_match.get("category_name"),
                'match_type': best_match.get("match_type", "NONE"),
                'llm_explanation': best_match.get("explanation", ""),
                'field_match_scores': field_scores,
                'matched_terms': {
                    'technical_terms': technical_terms,
                    'capabilities': capabilities
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                'llm_score': 0.0,
                'category_name': None,
                'match_type': "ERROR",
                'llm_explanation': f"Error during LLM analysis: {str(e)}",
                'field_match_scores': {},
                'matched_terms': {
                    'technical_terms': [],
                    'capabilities': []
                }
            }

    def _format_use_case_text(self, use_case: Dict[str, Any]) -> str:
        """Format use case data for LLM analysis."""
        sections = []
        
        # Add name
        if use_case.get('name'):
            sections.append(f"Name: {use_case['name']}")
            
        # Add description
        if use_case.get('description'):
            sections.append(f"Description: {use_case['description']}")
            
        # Add purpose and benefits
        if use_case.get('purpose_benefits'):
            sections.append(f"Purpose and Benefits: {use_case['purpose_benefits']}")
            
        # Add outputs
        if use_case.get('outputs'):
            sections.append(f"Outputs: {use_case['outputs']}")
            
        # Add topic area
        if use_case.get('topic_area'):
            sections.append(f"Topic Area: {use_case['topic_area']}")
            
        # Add stage
        if use_case.get('stage'):
            sections.append(f"Stage: {use_case['stage']}")
            
        # Add impact type
        if use_case.get('impact_type'):
            sections.append(f"Impact Type: {use_case['impact_type']}")
            
        # Add development method
        if use_case.get('dev_method'):
            sections.append(f"Development Method: {use_case['dev_method']}")
            
        # Add system name
        if use_case.get('system_name'):
            sections.append(f"System Name: {use_case['system_name']}")
            
        return "\n\n".join(sections)

    def _combine_analysis_results(self, results: Dict[str, Any]) -> None:
        """Combine results from different analysis methods and determine match types"""
        # Get individual scores
        keyword_score = results['keyword_score']
        semantic_score = results['semantic_score']
        llm_score = results['llm_score']
        
        # Get detailed field scores if available
        field_scores = results.get('field_match_scores', {})
        
        # Calculate enhanced confidence score using all available metrics
        confidence_components = {
            'keyword': keyword_score * 0.20,
            'semantic': semantic_score * 0.20,
            'llm': llm_score * 0.15,
            'technical': field_scores.get('technical_alignment', 0.0) * 0.10,
            'business': field_scores.get('business_alignment', 0.0) * 0.10,
            'implementation': field_scores.get('implementation_fit', 0.0) * 0.05,
            'capability': field_scores.get('capability_coverage', 0.0) * 0.05,
            'semantic_relevance': field_scores.get('semantic_relevance', 0.0) * 0.05,
            'keyword_relevance': field_scores.get('keyword_relevance', 0.0) * 0.05,
            'context_alignment': field_scores.get('context_alignment', 0.0) * 0.05
        }
        
        # Calculate weighted confidence score
        combined_score = sum(confidence_components.values())
        
        # Apply method agreement boost
        method_agreement = sum(1 for score in [
            keyword_score,
            semantic_score,
            llm_score,
            field_scores.get('technical_alignment', 0.0),
            field_scores.get('business_alignment', 0.0)
        ] if score > 0.4)
        
        if method_agreement > 1:
            combined_score *= (1 + (method_agreement - 1) * 0.1)  # 10% boost per agreeing method
            
        # Determine match type based on combined score and individual components
        if (combined_score > 0.45 and 
            min(keyword_score, semantic_score) > 0.3 and 
            field_scores.get('technical_alignment', 0.0) > 0.4 and
            field_scores.get('context_alignment', 0.0) > 0.35):
            # Strong match with good technical alignment and context
            results['match_type'] = 'PRIMARY'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 1
            
        elif (combined_score > 0.35 or 
              max(keyword_score, semantic_score) > 0.4 or 
              field_scores.get('business_alignment', 0.0) > 0.5 or
              field_scores.get('semantic_relevance', 0.0) > 0.45):
            # Good business alignment or strong individual scores
            results['match_type'] = 'SUPPORTING'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 2
            
        elif (combined_score > 0.25 or 
              any(score > 0.3 for score in [
                  keyword_score,
                  semantic_score,
                  llm_score,
                  field_scores.get('semantic_relevance', 0.0),
                  field_scores.get('context_alignment', 0.0)
              ])):
            # Some relevance detected
            results['match_type'] = 'RELATED'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 3
            
        else:
            results['match_type'] = 'NONE'
            results['match_method'] = 'INSUFFICIENT_CONFIDENCE'
            results['match_rank'] = 0
        
        # Update confidence score
        results['confidence'] = round(combined_score, 3)
        
        # Set category name based on match type and confidence
        if results['match_type'] != 'NONE':
            results['category_name'] = results.get('_best_match_category')
            results['relationship_type'] = 'USES_TECHNOLOGY'
        else:
            results['category_name'] = None
            results['relationship_type'] = 'NO_MATCH'
        
        # Enhance match details with semantic concepts if available
        term_match_details = results.get('term_match_details', {})
        if 'semantic_concepts' in term_match_details:
            semantic_concepts = term_match_details['semantic_concepts']
            # Add high-relevance concepts to matched terms
            for concept_type in ['technical_concepts', 'business_concepts']:
                for concept in semantic_concepts.get(concept_type, []):
                    if concept['relevance'] > 0.7:  # Only include high-relevance concepts
                        term_match_details.setdefault('context_matches', []).append(
                            f"{concept['concept']} ({concept['relevance']:.2f})"
                        )
        
        # Update v2.2 schema properties
        results.update({
            'field_match_scores': field_scores,
            'term_match_details': term_match_details,
            'analysis_version': 'v2.2',
            'review_status': 'PENDING',
            'false_positive': False,
            'manual_override': False
        })
        
        # Log combination details for analysis
        self.logger.debug(
            f"Score combination: keyword={keyword_score:.3f}, "
            f"semantic={semantic_score:.3f}, "
            f"llm={llm_score:.3f}, "
            f"combined={combined_score:.3f}, "
            f"match_type={results['match_type']}, "
            f"method={results['match_method']}"
        )

    def verify_match(self, use_case_text: str, category_name: str, match_type: str) -> float:
        """Verify a potential match using LLM."""
        try:
            response = self.llm_analyzer.verify_match(use_case_text, category_name, match_type)
            if isinstance(response, str):
                response_json = json.loads(response)
                confidence = float(response_json.get('confidence', 0.0))
                return confidence
            return 0.0
        except Exception as e:
            self.logger.error(f"LLM verification failed: {str(e)}")
            return 0.0

    def analyze_unmatched_case(self, use_case_text: str) -> Dict[str, Any]:
        """Analyze an unmatched use case using LLM."""
        try:
            response = self.llm_analyzer.analyze_unmatched(use_case_text)
            if isinstance(response, str):
                response_json = json.loads(response)
                return {
                    'reason': response_json.get('reason', 'Unknown'),
                    'reason_category': response_json.get('reason_category', 'Unknown'),
                    'suggestions': response_json.get('suggestions', [])
                }
            return {
                'reason': 'Error during analysis',
                'reason_category': 'Error',
                'suggestions': []
            }
        except Exception as e:
            self.logger.error(f"Error analyzing unmatched case: {str(e)}")
            return {
                'reason': 'Error during analysis',
                'reason_category': 'Error',
                'suggestions': []
            }

    def _format_categories_for_prompt(self) -> str:
        """Format AI categories for LLM prompt"""
        formatted_categories = []
        for category in self.categories.values():
            formatted = f"""
Category: {category.name}
Definition: {category.definition}
Keywords: {', '.join(category.keywords) if category.keywords else 'N/A'}
Capabilities: {', '.join(category.capabilities) if category.capabilities else 'N/A'}
Business Terms: {', '.join(category.business_language) if category.business_language else 'N/A'}
Maturity Level: {category.maturity_level}
"""
            formatted_categories.append(formatted)
        return "\n".join(formatted_categories)

    async def analyze_no_match(self, use_case: Dict[str, Any], current_category: Optional[str] = None) -> Dict[str, Any]:
        """Analyze cases with no strong matches using LLM."""
        try:
            # Format use case text
            use_case_text = self._format_use_case_text(use_case)
            
            # Get LLM analysis
            llm_results = await self.llm_analyzer.analyze_no_match(use_case_text, current_category)
            
            # Format results
            return {
                'category': llm_results.get('suggested_category'),
                'confidence': llm_results.get('confidence', 0.0),
                'explanation': llm_results.get('reasoning', ''),
                'suggestions': llm_results.get('suggestions', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                'category': None,
                'confidence': 0.0,
                'explanation': f"Error during analysis: {str(e)}",
                'suggestions': []
            }

    async def get_random_use_cases(self, num_cases: int) -> List[Dict]:
        """Get random use cases from Neo4j that haven't been classified yet."""
        try:
            async with self.driver.session() as session:
                # Try to get unclassified use cases first
                query = """
                MATCH (u:UseCase)
                WHERE NOT EXISTS((u)-[:CLASSIFIED_AS]->(:AIClassification))
                WITH u, rand() as r
                ORDER BY r
                LIMIT $limit
                RETURN {
                    id: u.id,
                    name: u.name,
                    description: u.description,
                    purpose_benefits: u.purpose_benefits,
                    outputs: u.outputs,
                    topic_area: u.topic_area,
                    stage: u.stage,
                    impact_type: u.impact_type,
                    dev_method: u.dev_method,
                    system_name: u.system_name
                } as use_case
                """
                result = await session.run(query, limit=num_cases)
                use_cases = [record["use_case"] async for record in result]
                
                # If we don't have enough unclassified cases, get any random cases
                if len(use_cases) < num_cases:
                    remaining = num_cases - len(use_cases)
                    query = """
                    MATCH (u:UseCase)
                    WITH u, rand() as r
                    ORDER BY r
                    LIMIT $limit
                    RETURN {
                        id: u.id,
                        name: u.name,
                        description: u.description,
                        purpose_benefits: u.purpose_benefits,
                        outputs: u.outputs,
                        topic_area: u.topic_area,
                        stage: u.stage,
                        impact_type: u.impact_type,
                        dev_method: u.dev_method,
                        system_name: u.system_name
                    } as use_case
                    """
                    result = await session.run(query, limit=remaining)
                    additional_cases = [record["use_case"] async for record in result]
                    use_cases.extend(additional_cases)
                
                return use_cases
                
        except Exception as e:
            logger.error(f"Error fetching random use cases: {str(e)}")
            return [] 