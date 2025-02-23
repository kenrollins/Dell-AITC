"""
Dell-AITC LLM Analyzer Service
Handles LLM-based analysis for AI technology classification using Ollama.
"""

import logging
from typing import Dict, Any, List, Optional
import httpx
import json
from datetime import datetime
import uuid
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """LLM-based analyzer for AI technology classification using Ollama"""
    
    def __init__(self, model: str = "phi4:latest", base_url: str = "http://localhost:11434"):
        """Initialize LLM analyzer"""
        self.client = None
        self.categories = {}
        self.base_url = base_url
        self.model = model
        
    async def initialize(self):
        """Initialize Ollama client"""
        # Initialize httpx client for Ollama
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0
        )
        logger.info(f"Initialized Ollama client with base URL: {self.base_url}")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()

    async def _cleanup_existing_classifications(self, use_case_id: str, session) -> None:
        """Remove existing classifications for a use case before adding new ones."""
        try:
            # Delete existing CLASSIFIED_AS relationships
            await session.run("""
            MATCH (u:UseCase {id: $use_case_id})-[r:CLASSIFIED_AS]->()
            DELETE r
            """, {"use_case_id": use_case_id})
            
            # Delete existing NoMatchAnalysis nodes and relationships
            await session.run("""
            MATCH (u:UseCase {id: $use_case_id})-[r:HAS_ANALYSIS]->(n:NoMatchAnalysis)
            DELETE r, n
            """, {"use_case_id": use_case_id})
            
            logger.info(f"Cleaned up existing classifications for use case {use_case_id}")
        except Exception as e:
            logger.error(f"Error cleaning up existing classifications: {str(e)}")
            raise
            
    async def analyze_use_case(self, use_case: Dict[str, Any], categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a use case and determine matching AI technology categories."""
        try:
            # Create the classification prompt
            prompt = self._create_classification_prompt(use_case, categories)
            
            logger.info(f"Making request to Ollama API with model {self.model}")
            
            # Get LLM analysis using Ollama
            request_data = {
                "model": self.model,
                "prompt": f"""You are an expert AI technology classifier for federal use cases.
Your task is to analyze federal use cases and match them to the most appropriate AI technology categories
based on technical alignment, capabilities, and implementation patterns.

{prompt}

Remember to respond ONLY with valid JSON matching the format specified in the prompt.""",
                "stream": False
            }
            logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
            
            response = await self.client.post("api/generate", json=request_data)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            logger.debug(f"Raw Ollama response: {json.dumps(result, indent=2)}")
            
            # Extract the response text and parse as JSON
            try:
                response_text = result["response"]
                # Handle markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1]
                if "```" in response_text:
                    response_text = response_text.split("```")[0]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse Ollama response: {str(e)}")
                logger.error(f"Raw response: {result}")
                logger.error(f"Extracted text: {response_text if 'response_text' in locals() else 'N/A'}")
                return {
                    "error": "Failed to parse response",
                    "analyzed_at": datetime.now().isoformat(),
                    "use_case_id": use_case.get("id", "unknown"),
                    "no_match_analysis": {
                        "reason": "Failed to parse LLM response",
                        "technical_gaps": [],
                        "suggested_focus": "Analysis could not be completed due to response parsing error"
                    }
                }
            
            # Add metadata
            result["analyzed_at"] = datetime.now().isoformat()
            result["use_case_id"] = use_case.get("id", "unknown")
            
            return result
            
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Ollama API: {str(e)}")
            return {
                "error": f"Connection error: {str(e)}",
                "analyzed_at": datetime.now().isoformat(),
                "use_case_id": use_case.get("id", "unknown"),
                "no_match_analysis": {
                    "reason": f"Connection failed: {str(e)}",
                    "technical_gaps": [],
                    "suggested_focus": "Analysis could not be completed due to connection error"
                }
            }
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error: {str(e)}")
            return {
                "error": f"Timeout error: {str(e)}",
                "analyzed_at": datetime.now().isoformat(),
                "use_case_id": use_case.get("id", "unknown"),
                "no_match_analysis": {
                    "reason": f"Request timed out: {str(e)}",
                    "technical_gaps": [],
                    "suggested_focus": "Analysis could not be completed due to timeout"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}", exc_info=True)
            return {
                "error": f"Error in LLM analysis: {str(e)}",
                "analyzed_at": datetime.now().isoformat(),
                "use_case_id": use_case.get("id", "unknown"),
                "no_match_analysis": {
                    "reason": f"Analysis failed: {str(e)}",
                    "technical_gaps": [],
                    "suggested_focus": "Analysis could not be completed due to unexpected error"
                }
            }
            
    def _create_classification_prompt(self, use_case: Dict[str, Any], categories: List[Dict[str, Any]]) -> str:
        """Create a detailed prompt for classification analysis."""
        
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
Technical Keywords: {', '.join(cat.get('keywords', []))}
Core Capabilities: {', '.join(cat.get('capabilities', []))}
"""

        # Create the full prompt
        return f"""{use_case_text}

{categories_text}

Analyze this federal use case and determine the most appropriate AI technology category matches.

Provide your analysis in JSON format:
{{
    "primary_match": {{
        "category": string,  // Name of the single best matching category
        "confidence": float,  // Between 0.0-1.0
        "reasoning": string  // 2-3 sentences explaining the match, including:
                           // 1. Technical alignment with category definition
                           // 2. Specific capabilities that match
                           // 3. Implementation considerations
    }},
    "supporting_matches": [
        {{
            "category": string,
            "confidence": float,
            "reasoning": string  // 2-3 sentences explaining:
                               // 1. How this complements the primary category
                               // 2. Specific technical integration points
                               // 3. Additional capabilities provided
        }}
    ],
    "related_matches": [
        {{
            "category": string,
            "confidence": float,
            "reasoning": string  // 1-2 sentences explaining:
                               // 1. Tangential relevance to solution
                               // 2. Potential future integration points
        }}
    ],
    "no_match_analysis": {{  // Only if no good matches found
        "reason": string,  // 2-3 sentences explaining why no categories match well
        "technical_gaps": [string],  // List specific technical capabilities missing
        "suggested_focus": string  // 1-2 sentences suggesting technical direction
    }}
}}

Classification Guidelines:
1. Primary Match (Required if any good match exists):
   - Must be the SINGLE most appropriate category
   - Should have confidence > 0.8
   - Must align with core technical requirements
   - Provide specific technical justification with examples from use case

2. Supporting Matches (Optional, up to 2):
   - Categories that complement the primary category
   - Should have confidence > 0.6
   - Explain specific technical integration points
   - Detail how they enhance the primary category's capabilities

3. Related Matches (Optional):
   - Categories with tangential relevance
   - Should have confidence > 0.4
   - Explain potential technical synergies
   - Note specific future integration possibilities

4. No Match Analysis (Required if no primary match):
   - Explain specific technical gaps
   - List missing capabilities
   - Suggest concrete technical focus areas
   - Note any partial matches

Analysis Focus:
- Technical alignment with category definitions
- Required capabilities and components
- Implementation patterns and maturity
- Integration requirements
- Development stage compatibility

For each match, ensure reasoning:
1. References specific aspects of the use case
2. Cites relevant technical capabilities
3. Considers implementation context
4. Explains technical integration points
5. Notes any important caveats or considerations""" 

    async def save_classification_results(self, use_case_id: str, results: Dict[str, Any], driver) -> None:
        """Save classification results to Neo4j.
        
        Args:
            use_case_id: ID of the use case being classified
            results: Classification results from analyze_use_case
            driver: Neo4j driver instance
        """
        try:
            async with driver.session() as session:
                # First cleanup existing classifications
                await self._cleanup_existing_classifications(use_case_id, session)
                
                # Handle primary match
                if primary := results.get("primary_match"):
                    await session.run("""
                    MATCH (u:UseCase {id: $use_case_id})
                    MATCH (c:AICategory {name: $category_name})
                    CREATE (u)-[r:CLASSIFIED_AS {
                        id: $classification_id,
                        match_type: 'PRIMARY',
                        confidence: $confidence,
                        reasoning: $reasoning,
                        classified_at: $timestamp,
                        classified_by: 'LLM_ANALYZER',
                        analysis_method: 'LLM'
                    }]->(c)
                    """, {
                        "use_case_id": use_case_id,
                        "category_name": primary["category"],
                        "classification_id": str(uuid.uuid4()),
                        "confidence": primary["confidence"],
                        "reasoning": primary["reasoning"],
                        "timestamp": datetime.now().isoformat()
                    })

                # Handle supporting matches
                for match in results.get("supporting_matches", []):
                    await session.run("""
                    MATCH (u:UseCase {id: $use_case_id})
                    MATCH (c:AICategory {name: $category_name})
                    CREATE (u)-[r:CLASSIFIED_AS {
                        id: $classification_id,
                        match_type: 'SUPPORTING',
                        confidence: $confidence,
                        reasoning: $reasoning,
                        classified_at: $timestamp,
                        classified_by: 'LLM_ANALYZER',
                        analysis_method: 'LLM'
                    }]->(c)
                    """, {
                        "use_case_id": use_case_id,
                        "category_name": match["category"],
                        "classification_id": str(uuid.uuid4()),
                        "confidence": match["confidence"],
                        "reasoning": match["reasoning"],
                        "timestamp": datetime.now().isoformat()
                    })

                # Handle related matches
                for match in results.get("related_matches", []):
                    await session.run("""
                    MATCH (u:UseCase {id: $use_case_id})
                    MATCH (c:AICategory {name: $category_name})
                    CREATE (u)-[r:CLASSIFIED_AS {
                        id: $classification_id,
                        match_type: 'RELATED',
                        confidence: $confidence,
                        reasoning: $reasoning,
                        classified_at: $timestamp,
                        classified_by: 'LLM_ANALYZER',
                        analysis_method: 'LLM'
                    }]->(c)
                    """, {
                        "use_case_id": use_case_id,
                        "category_name": match["category"],
                        "classification_id": str(uuid.uuid4()),
                        "confidence": match["confidence"],
                        "reasoning": match["reasoning"],
                        "timestamp": datetime.now().isoformat()
                    })

                # Only create NoMatchAnalysis if there are no matches at all
                if not results.get("primary_match") and not results.get("supporting_matches"):
                    if no_match := results.get("no_match_analysis"):
                        await session.run("""
                        MATCH (u:UseCase {id: $use_case_id})
                        CREATE (n:NoMatchAnalysis {
                            id: $analysis_id,
                            reason: $reason,
                            technical_gaps: $gaps,
                            suggested_focus: $focus,
                            created_at: $timestamp,
                            analyzed_by: 'LLM_ANALYZER',
                            status: 'NEW'
                        })
                        CREATE (u)-[r:HAS_ANALYSIS {created_at: $timestamp}]->(n)
                        """, {
                            "use_case_id": use_case_id,
                            "analysis_id": str(uuid.uuid4()),
                            "reason": no_match["reason"],
                            "gaps": no_match["technical_gaps"],
                            "focus": no_match["suggested_focus"],
                            "timestamp": datetime.now().isoformat()
                        })

        except Exception as e:
            logger.error(f"Error saving classification results: {str(e)}")
            raise 