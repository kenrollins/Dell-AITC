from typing import List, Dict, Optional
import logging
from neo4j import AsyncGraphDatabase
from ...config import get_settings

class Classifier:
    def __init__(self):
        """Initialize the classifier with Neo4j connection."""
        self.logger = logging.getLogger(__name__)
        self.driver = None
        
    async def initialize(self):
        """Initialize the classifier by setting up the Neo4j connection."""
        self.logger.info("Initializing base classifier")
        settings = get_settings()
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def close(self):
        """Close the Neo4j driver connection."""
        await self.driver.close()

    async def get_unclassified_use_cases(self, limit: Optional[int] = None) -> List[Dict]:
        """Get use cases that haven't been classified yet.
        
        Args:
            limit: Optional maximum number of use cases to return
            
        Returns:
            List of use case dictionaries
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
            status: u.status
        } as use_case
        """
        if limit:
            query += f" LIMIT {limit}"
            
        async with self.driver.session() as session:
            result = await session.run(query)
            records = await result.data()
            return [record["use_case"] for record in records] 

    async def _perform_llm_analysis(self, use_case: dict, category: dict) -> float:
        """
        Perform LLM analysis to determine if a use case matches a category.
        Returns a score between 0 and 1.
        """
        try:
            # Prepare the prompt
            prompt = self._prepare_llm_prompt(use_case, category)
            
            # Get LLM analysis
            llm_response = await self.llm_service.analyze_text(prompt)
            
            # Extract score based on match type and confidence
            match_type = llm_response.get("match_type", "NO_MATCH")
            confidence = float(llm_response.get("confidence", 0.0))
            
            # Calculate final score based on match type and confidence
            match_type_scores = {
                "PRIMARY": 1.0,
                "SUPPORTING": 0.7,
                "RELATED": 0.4,
                "NO_MATCH": 0.0,
                "ERROR": 0.0
            }
            
            base_score = match_type_scores.get(match_type, 0.0)
            final_score = base_score * confidence
            
            # Log the analysis details
            self.logger.info(f"LLM Analysis for use case {use_case.get('id')} and category {category.get('name')}:")
            self.logger.info(f"Match Type: {match_type}")
            self.logger.info(f"Confidence: {confidence}")
            self.logger.info(f"Final Score: {final_score}")
            self.logger.info(f"Justification: {llm_response.get('justification', 'No justification provided')}")
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating LLM score: {str(e)}", exc_info=True)
            return 0.0

    def _prepare_llm_prompt(self, use_case: dict, category: dict) -> str:
        """
        Prepare the prompt for LLM analysis.
        """
        # Extract use case details
        use_case_desc = use_case.get("description", "")
        use_case_title = use_case.get("title", "")
        use_case_stage = use_case.get("stage", "")
        use_case_topic = use_case.get("topic_area", "")
        use_case_impact = use_case.get("impact_type", "")
        
        # Extract category details
        category_name = category.get("name", "")
        category_def = category.get("definition", "")
        
        # Build the prompt
        prompt = f"""Analyze if the following federal use case matches the AI technology category.

Use Case Details:
Title: {use_case_title}
Description: {use_case_desc}
Development Stage: {use_case_stage}
Topic Area: {use_case_topic}
Impact Type: {use_case_impact}

AI Technology Category:
Name: {category_name}
Definition: {category_def}

Analyze if this use case matches the AI technology category. Consider:
1. Direct alignment with category definition
2. Technical requirements and capabilities
3. Business objectives and outcomes
4. Development stage and maturity
5. Potential impact and benefits

Provide your analysis in a structured format indicating:
- Match type (PRIMARY/SUPPORTING/RELATED/NO_MATCH)
- Confidence level (0-1)
- Detailed justification
- Suggestions for improvement
- Relevant technical and business terms identified
"""
        return prompt 