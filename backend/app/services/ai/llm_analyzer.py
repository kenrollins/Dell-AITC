"""
Dell-AITC LLM Analyzer Service
Handles LLM-based analysis for AI technology classification.
"""

import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from ...config import get_settings
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """LLM-based analyzer for AI technology classification"""
    
    def __init__(self):
        """Initialize LLM analyzer"""
        self.client = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize OpenAI client"""
        settings = get_settings()
        
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()
            
    def _create_classification_prompt(self, use_case: Dict[str, Any], categories: List[Dict[str, Any]]) -> str:
        """Create a detailed prompt for classification
        
        Args:
            use_case: Use case to classify
            categories: List of AI technology categories
            
        Returns:
            Formatted prompt string
        """
        # Format use case details
        use_case_text = f"""Use Case Details:
Name: {use_case.get('name', 'Untitled')}
Description: {use_case.get('description', 'No description')}
Purpose & Benefits: {use_case.get('purpose_benefits', 'Not specified')}
Outputs: {use_case.get('outputs', 'Not specified')}
Development Stage: {use_case.get('dev_stage', 'Unknown')}
Development Method: {use_case.get('dev_method', 'Unknown')}"""

        # Format categories with their details
        categories_text = "Available AI Technology Categories:\n\n"
        for idx, cat in enumerate(categories, 1):
            categories_text += f"""Category {idx}: {cat.get('name', 'Unnamed')}
Definition: {cat.get('category_definition', 'No definition')}
Zone: {cat.get('zone', 'Unassigned')}
Maturity Level: {cat.get('maturity_level', 'Unknown')}
Technical Keywords: {', '.join(kw.get('name', '') for kw in cat.get('keywords', []))}
Business Terms: {', '.join(term.get('name', '') for term in cat.get('business_terms', []))}
Capabilities: {', '.join(cat.get('capabilities', []))}
"""

        # Create the full prompt
        return f"""You are an expert AI technology classifier for federal use cases. Your task is to analyze a federal use case and determine which AI technology categories best match its requirements and characteristics.

{use_case_text}

{categories_text}

Analysis Guidelines:
1. Primary Match: Identify the SINGLE most appropriate category that best matches the use case's core technology needs (confidence > 0.8)
2. Supporting Matches: Identify up to 2 categories that complement the primary category (confidence > 0.6)
3. Related Matches: Note any categories that have tangential relevance (confidence > 0.4)

For each potential match, consider:
- Direct alignment with category definition
- Technical keyword matches
- Business term relevance
- Capability requirements
- Zone compatibility
- Maturity level appropriateness

If no categories match well (confidence < 0.4), provide a detailed analysis of why.

Provide your analysis in JSON format:
{{
    "primary_match": {{
        "category": "Category name",
        "confidence": float,  // 0.0-1.0
        "reasoning": "Detailed explanation of why this category is the best match",
        "technical_alignment": {{
            "matched_terms": ["list of matched technical terms"],
            "matched_capabilities": ["list of matched capabilities"],
            "zone_compatibility": "explanation of zone fit"
        }}
    }},
    "supporting_matches": [
        {{
            "category": "Category name",
            "confidence": float,
            "reasoning": "Why this category complements the primary match"
        }}
    ],
    "related_matches": [
        {{
            "category": "Category name",
            "confidence": float,
            "reasoning": "Why this category is tangentially relevant"
        }}
    ],
    "no_match_analysis": {{  // Only if no good matches found
        "reason": "Primary reason for no matches",
        "technical_gaps": ["List of technical gaps identified"],
        "suggested_focus": "Suggested technical direction",
        "improvement_suggestions": ["List of suggestions to improve matching"]
    }}
}}"""

    async def analyze_use_case(self, use_case: Dict[str, Any], categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a use case against available categories
        
        Args:
            use_case: Use case to analyze
            categories: List of AI technology categories
            
        Returns:
            Analysis results including matches and reasoning
        """
        try:
            # Create the prompt
            prompt = self._create_classification_prompt(use_case, categories)
            
            # Get LLM analysis
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use standard OpenAI model
                messages=[
                    {"role": "system", "content": "You are an expert AI technology classifier specializing in federal use cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result["analyzed_at"] = datetime.now().isoformat()
            result["use_case_id"] = use_case.get("id", "unknown")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                "error": str(e),
                "analyzed_at": datetime.now().isoformat(),
                "use_case_id": use_case.get("id", "unknown"),
                "no_match_analysis": {
                    "reason": f"Analysis failed: {str(e)}",
                    "technical_gaps": [],
                    "suggested_focus": "Analysis could not be completed",
                    "improvement_suggestions": ["Retry analysis"]
                }
            } 