from datetime import datetime
from typing import Dict, Any, List
import uuid

class FedUseCaseClassifier:
    def __init__(self, db_service, llm_service):
        self.db = db_service
        self.llm = llm_service
        self.category_cache = {}

    async def _get_category_details(self, category_id: str) -> Dict[str, Any]:
        """Get detailed category information from Neo4j."""
        query = """
        MATCH (c:AICategory {id: $category_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
        WITH c, z, 
             collect(DISTINCT {
                name: k.name,
                type: coalesce(k.type, 'technical'),
                relevance_score: coalesce(k.relevance_score, 0.5)
             }) as keywords
        RETURN {
            id: c.id,
            name: coalesce(c.name, 'Unnamed Category'),
            category_definition: c.category_definition,
            maturity_level: coalesce(c.maturity_level, 'Unknown'),
            status: coalesce(c.status, 'Active'),
            zone: coalesce(z.name, 'Unassigned'),
            keywords: [kw IN keywords WHERE kw.type = 'technical'],
            business_terms: [kw IN keywords WHERE kw.type = 'business'],
            created_at: toString(datetime()),
            last_updated: toString(datetime()),
            version: coalesce(c.version, '1.0')
        } as category
        """
        
        result = await self.db.run_query(query, {"category_id": category_id})
        if not result:
            raise ValueError(f"Category with ID {category_id} not found")
            
        return result[0]['category']

    async def classify_use_case(self, use_case: Dict[str, Any], category: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a use case against an AI category."""
        category_details = await self._get_category_details(category['id'])
        
        # Create enhanced prompt with category details
        prompt = f"""
        Use Case:
        Name: {use_case['name']}
        Description: {use_case['description']}
        Purpose & Benefits: {use_case['purpose_benefits']}
        Outputs: {use_case['outputs']}
        Development Stage: {use_case.get('dev_stage', 'Unknown')}
        Development Method: {use_case.get('dev_method', 'Unknown')}
        
        AI Category:
        Name: {category_details['name']}
        Definition: {category_details['category_definition']}
        Technical Keywords: {', '.join(category_details['keywords'])}
        Business Terms: {', '.join(category_details.get('business_terms', []))}
        Zone: {category_details['zone']}
        Maturity Level: {category_details['maturity_level']}
        
        Analyze if this use case belongs to this AI category. Consider:
        1. Technical alignment with keywords
        2. Business alignment with terms
        3. Maturity and development stage compatibility
        4. Overall fit with category definition
        
        Provide:
        1. Match type (STRONG_MATCH, PARTIAL_MATCH, or NO_MATCH)
        2. Confidence score (0.0-1.0)
        3. Brief justification
        4. Improvement suggestions if any
        """
        
        # Get LLM analysis
        analysis = await self.llm.analyze_text(prompt)
        
        # Extract key information (implement based on LLM response format)
        match_type = self._determine_match_type(analysis)
        confidence = self._calculate_confidence(analysis)
        justification = self._extract_justification(analysis)
        suggestions = self._extract_suggestions(analysis)
        
        return {
            "use_case_id": use_case['id'],
            "category_id": category['id'],
            "match_type": match_type,
            "confidence": confidence,
            "field_match_scores": {
                "technical_alignment": self._calculate_technical_score(use_case, category_details),
                "business_alignment": self._calculate_business_score(use_case, category_details),
                "maturity_compatibility": self._calculate_maturity_score(use_case, category_details)
            },
            "matched_terms": {
                "technical": self._find_matching_keywords(use_case, category_details['keywords']),
                "business": self._find_matching_terms(use_case, category_details.get('business_terms', []))
            },
            "justification": justification,
            "improvement_suggestions": suggestions,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _filter_keywords(self, keywords: List[str], keyword_type: str) -> List[str]:
        """Filter keywords by type (technical or business)."""
        # In this implementation, we're using a simple heuristic
        # Technical keywords typically contain specific technical terms
        technical_indicators = {'ai', 'ml', 'api', 'data', 'model', 'algorithm', 'system', 'neural',
                              'cloud', 'platform', 'framework', 'analytics', 'automation'}
        
        if keyword_type == 'technical':
            return [k for k in keywords if any(ind in k.lower() for ind in technical_indicators)]
        else:
            return [k for k in keywords if not any(ind in k.lower() for ind in technical_indicators)]

    def _find_capability_matches(self, use_case: dict, capabilities: List[str]) -> List[str]:
        """Find matching capabilities between use case and category."""
        matches = []
        use_case_text = f"{use_case.get('description', '')} {use_case.get('purpose_benefits', '')}"
        
        for capability in capabilities:
            if capability.lower() in use_case_text.lower():
                matches.append(capability)
        
        return matches

    def _create_no_match_response(self, reason: str) -> dict:
        """Create a standardized no-match response."""
        return {
            "match_type": "NONE",
            "confidence": 0.0,
            "field_match_scores": {
                "technical_alignment": 0.0,
                "keyword_relevance": 0.0,
                "capability_match": 0.0,
                "zone_compatibility": 0.0,
                "implementation_fit": 0.0
            },
            "matched_terms": {
                "technical_terms": [],
                "business_terms": [],
                "capabilities": []
            },
            "justification": reason,
            "improvement_suggestions": ["Category details not available for analysis"]
        }

    def _determine_match_type(self, analysis: Dict[str, Any]) -> str:
        """Determine the match type from LLM analysis."""
        if not analysis or 'choices' not in analysis:
            return 'NO_MATCH'
        
        content = analysis['choices'][0]['message']['content']
        if 'STRONG_MATCH' in content:
            return 'STRONG_MATCH'
        elif 'PARTIAL_MATCH' in content:
            return 'PARTIAL_MATCH'
        return 'NO_MATCH'

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score from LLM analysis."""
        if not analysis or 'choices' not in analysis:
            return 0.0
        
        content = analysis['choices'][0]['message']['content']
        try:
            # Extract confidence score from content (format: "Confidence score: X.XX")
            score_text = content.split('Confidence score:')[1].split('\n')[0].strip()
            return float(score_text)
        except (IndexError, ValueError):
            return 0.0

    def _extract_justification(self, analysis: Dict[str, Any]) -> str:
        """Extract justification from LLM analysis."""
        if not analysis or 'choices' not in analysis:
            return "No justification available"
        
        content = analysis['choices'][0]['message']['content']
        try:
            # Extract justification (format: "Justification: ...")
            return content.split('Justification:')[1].split('\n')[0].strip()
        except IndexError:
            return "No justification available"

    def _extract_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract improvement suggestions from LLM analysis."""
        if not analysis or 'choices' not in analysis:
            return []
        
        content = analysis['choices'][0]['message']['content']
        try:
            # Extract suggestions (format: "Improvement suggestions: 1. ... 2. ...")
            suggestions_text = content.split('Improvement suggestions:')[1].strip()
            return [s.strip() for s in suggestions_text.split('\n') if s.strip()]
        except IndexError:
            return []

    def _calculate_technical_score(self, use_case: Dict[str, Any], category: Dict[str, Any]) -> float:
        """Calculate technical alignment score."""
        keywords = category['keywords']
        if not keywords:
            return 0.0
        
        text = f"{use_case['description']} {use_case['purpose_benefits']} {use_case['outputs']}"
        text = text.lower()
        
        # Calculate weighted score based on keyword matches and their scores
        total_score = 0.0
        max_possible = 0.0
        
        for kw in keywords:
            keyword = kw['name'].lower()
            weight = kw['relevance_score']
            max_possible += weight
            
            if keyword in text:
                total_score += weight
            # Partial word matches get partial credit
            elif any(keyword in word or word in keyword for word in text.split()):
                total_score += weight * 0.5
                
        return total_score / max_possible if max_possible > 0 else 0.0

    def _calculate_business_score(self, use_case: Dict[str, Any], category: Dict[str, Any]) -> float:
        """Calculate business alignment score."""
        terms = category.get('business_terms', [])
        if not terms:
            return 0.0
        
        text = f"{use_case['description']} {use_case['purpose_benefits']}"
        text = text.lower()
        
        # Calculate weighted score based on business term matches and their scores
        total_score = 0.0
        max_possible = 0.0
        
        for term in terms:
            business_term = term['name'].lower()
            weight = term['relevance_score']
            max_possible += weight
            
            if business_term in text:
                total_score += weight
            # Partial phrase matches get partial credit
            elif all(word in text for word in business_term.split()):
                total_score += weight * 0.75
                
        return total_score / max_possible if max_possible > 0 else 0.0

    def _calculate_maturity_score(self, use_case: Dict[str, Any], category: Dict[str, Any]) -> float:
        """Calculate maturity compatibility score."""
        maturity_levels = {
            'EMERGING': 0.2,
            'DEVELOPING': 0.4,
            'STABLE': 0.6,
            'MATURE': 0.8,
            'ESTABLISHED': 1.0
        }
        
        category_maturity = category.get('maturity_level', 'UNKNOWN').upper()
        use_case_stage = use_case.get('dev_stage', 'UNKNOWN').upper()
        
        if category_maturity == 'UNKNOWN' or use_case_stage == 'UNKNOWN':
            return 0.5
        
        category_score = maturity_levels.get(category_maturity, 0.5)
        use_case_score = maturity_levels.get(use_case_stage, 0.5)
        
        return 1.0 - abs(category_score - use_case_score)

    def _find_matching_keywords(self, use_case: Dict[str, Any], keywords: List[str]) -> List[str]:
        """Find matching technical keywords in use case text."""
        text = f"{use_case['description']} {use_case['purpose_benefits']} {use_case['outputs']}"
        return [kw for kw in keywords if kw.lower() in text.lower()]

    def _find_matching_terms(self, use_case: Dict[str, Any], terms: List[str]) -> List[str]:
        """Find matching business terms in use case text."""
        text = f"{use_case['description']} {use_case['purpose_benefits']}"
        return [term for term in terms if term.lower() in text.lower()] 