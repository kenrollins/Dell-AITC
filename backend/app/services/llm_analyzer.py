"""
LLM Analyzer Service for AI Technology Classification

This service uses LLM to:
1. Verify potential matches between use cases and AI categories
2. Analyze cases where no match was found
3. Suggest improvements to classification system
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
from dotenv import load_dotenv
from pathlib import Path
import httpx
from ..config import get_settings
from neo4j import AsyncGraphDatabase
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define official categories
OFFICIAL_CATEGORIES = [
    "AI Development & Operations (AIOps)",
    "Computer Vision & Media Analysis",
    "Cybersecurity & Threat Detection",
    "Data Integration & Management",
    "Decision Support & Optimization",
    "Edge AI & IoT",
    "Environmental & Geospatial AI",
    "Healthcare & Biotech AI",
    "Intelligent End-User Computing",
    "Multimodal AI Systems",
    "Natural Language Processing (NLP)",
    "Predictive & Pattern Analytics",
    "Process Automation & Robotics",
    "Responsible AI Systems"
]

class LLMAnalyzer:
    """Handles LLM-based analysis for AI technology classification."""
    
    def __init__(self):
        """Initialize with configuration."""
        self.settings = get_settings()
        self.neo4j_driver = None
        self.openai_client = None
        self.category_definitions = {}
        
        # Get retry and timeout settings from config
        self.max_retries = self.settings.openai_max_retries
        self.retry_delays = self.settings.openai_retry_delays
        self.api_timeout = self.settings.openai_timeout
        
    async def initialize(self):
        """Initialize connections and load category definitions."""
        # Initialize Neo4j connection
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password)
        )
        
        # Initialize OpenAI client with configured timeout
        self.openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            timeout=self.api_timeout
        )
        
        # Load category definitions
        await self._load_category_definitions()
        
    async def cleanup(self):
        """Cleanup connections and resources."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.openai_client:
            await self.openai_client.aclose()
        
    async def _load_category_definitions(self):
        """Load AI technology category definitions from Neo4j."""
        try:
            if not self.neo4j_driver:
                raise ValueError("Error: Neo4j connection not available")
                
            async with self.neo4j_driver.session() as session:
                query = """
                MATCH (c:AICategory)
                WHERE c.status = 'active'
                OPTIONAL MATCH (c)-[r1:HAS_KEYWORD]->(k1:Keyword)
                WHERE r1.type = 'technical'
                WITH c, collect(DISTINCT k1.name) as keywords
                OPTIONAL MATCH (c)-[r2:HAS_KEYWORD]->(k2:Keyword)
                WHERE r2.type = 'capability'
                WITH c, keywords, collect(DISTINCT k2.name) as capabilities
                OPTIONAL MATCH (c)-[r3:HAS_KEYWORD]->(k3:Keyword)
                WHERE r3.type = 'business_term'
                WITH c, keywords, capabilities, collect(DISTINCT k3.name) as business_terms
                OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
                RETURN {
                    name: c.name,
                    definition: c.category_definition,
                    keywords: keywords,
                    capabilities: capabilities,
                    business_language: business_terms,
                    maturity_level: c.maturity_level,
                    zone: z.name
                } as category
                """
                result = await session.run(query)
                async for record in result:
                    category = record["category"]
                    self.category_definitions[category["name"]] = category
        except Exception as e:
            raise RuntimeError(f"Error: Neo4j database error - {str(e)}")
        
    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Call OpenAI API with improved error handling and retries.
        Uses exponential backoff for retries.
        
        Args:
            messages: List of message dictionaries for the chat completion
            
        Returns:
            Properly formatted response dict with all required fields
        """
        last_error = None
        
        for retry_attempt in range(self.max_retries):
            try:
                response = await self.openai_client.post(
                    "/chat/completions",
                    json={
                        "model": self.settings.openai_model or "gpt-4",
                        "messages": messages,
                        "temperature": 0.3,
                        "response_format": { "type": "json_object" }
                    }
                )
                response.raise_for_status()
                
                # Parse response and ensure it's valid JSON
                try:
                    result = json.loads(response.json()["choices"][0]["message"]["content"])
                    return self._validate_openai_response(result)
                except (json.JSONDecodeError, KeyError) as e:
                    last_error = f"Failed to parse OpenAI response: {str(e)}"
                    logger.warning(f"Attempt {retry_attempt + 1}: {last_error}")
                    
            except (httpx.TimeoutException, httpx.ReadTimeout) as e:
                last_error = f"OpenAI API timeout: {str(e)}"
                logger.warning(f"Attempt {retry_attempt + 1}: {last_error}")
                
            except Exception as e:
                last_error = f"OpenAI API error: {str(e)}"
                logger.warning(f"Attempt {retry_attempt + 1}: {last_error}")
            
            # Don't sleep after the last attempt
            if retry_attempt < self.max_retries - 1:
                delay = self.retry_delays[retry_attempt]
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return self._get_default_response(f"Error after {self.max_retries} attempts: {last_error}")
    
    def _validate_openai_response(self, result: Dict) -> Dict:
        """Validate and format OpenAI API response."""
        return {
            "best_match": {
                "category_name": str(result.get("best_match", {}).get("category_name", "")),
                "confidence": float(result.get("best_match", {}).get("confidence", 0.0)),
                "match_type": str(result.get("best_match", {}).get("match_type", "NONE")),
                "reasoning": str(result.get("best_match", {}).get("reasoning", "No reasoning provided"))
            },
            "field_analysis": {
                "technical_alignment": float(result.get("field_analysis", {}).get("technical_alignment", 0.0)),
                "business_alignment": float(result.get("field_analysis", {}).get("business_alignment", 0.0)),
                "implementation_fit": float(result.get("field_analysis", {}).get("implementation_fit", 0.0)),
                "capability_coverage": float(result.get("field_analysis", {}).get("capability_coverage", 0.0)),
                "maturity_alignment": float(result.get("field_analysis", {}).get("maturity_alignment", 0.0))
            },
            "matched_terms": {
                "technical_terms": list(result.get("matched_terms", {}).get("technical_terms", [])),
                "business_terms": list(result.get("matched_terms", {}).get("business_terms", [])),
                "capabilities": list(result.get("matched_terms", {}).get("capabilities", []))
            },
            "alternative_matches": list(result.get("alternative_matches", []))
        }

    def _get_default_response(self, error_message: str) -> Dict:
        """Get a default response with all required fields."""
        return {
            "best_match": {
                "category_name": "",
                "confidence": 0.0,
                "match_type": "NONE",
                "reasoning": error_message
            },
            "field_analysis": {
                "technical_alignment": 0.0,
                "business_alignment": 0.0,
                "implementation_fit": 0.0,
                "capability_coverage": 0.0,
                "maturity_alignment": 0.0
            },
            "matched_terms": {
                "technical_terms": [],
                "business_terms": [],
                "capabilities": []
            },
            "alternative_matches": []
        }

    async def verify_match(
        self,
        use_case_text: str,
        category_name: str,
        match_type: str,
        confidence: float
    ) -> Dict:
        """
        Verify a potential match between a use case and an AI technology category.
        """
        # Validate category exists and is official
        if category_name not in self.category_definitions:
            raise ValueError(f"Unknown category: {category_name}")
        if category_name not in OFFICIAL_CATEGORIES:
            raise ValueError(f"Category {category_name} is not in the official list of categories")
            
        # Validate match type
        valid_match_types = ["PRIMARY", "SUPPORTING", "RELATED", "NONE"]
        if match_type not in valid_match_types:
            raise ValueError(f"Invalid match type: {match_type}. Must be one of {valid_match_types}")
            
        # Get category details
        category = self.category_definitions[category_name]
        
        # Safely get list properties with defaults
        keywords = category.get('keywords', [])
        if keywords is None:
            keywords = []
        elif isinstance(keywords, str):
            keywords = [keywords]
            
        capabilities = category.get('capabilities', [])
        if capabilities is None:
            capabilities = []
        elif isinstance(capabilities, str):
            capabilities = [capabilities]
            
        business_terms = category.get('business_language', [])
        if business_terms is None:
            business_terms = []
        elif isinstance(business_terms, str):
            business_terms = [business_terms]
        
        # Build enhanced prompt with official categories context
        messages = [
            {"role": "system", "content": f"""You are an expert AI technology analyst evaluating federal use cases against our official AI technology categories:

{chr(10).join(f'- {cat}' for cat in OFFICIAL_CATEGORIES)}

Your task is to verify if this use case truly matches the specified category based on detailed analysis.
Only validate matches against these official categories.

Provide your response in JSON format with the following structure:
{{
    "agrees": boolean,                // Whether you agree with the match
    "confidence": float,              // Your confidence in the assessment (0.0-1.0)
    "match_type": string,             // "PRIMARY", "SUPPORTING", "RELATED", or "NONE"
    "reasoning": string,              // Your detailed reasoning
    "field_match_scores": {{
        "technical_alignment": float,  // How well technical aspects align
        "business_alignment": float,   // How well business objectives align
        "implementation_fit": float,   // How well implementation approach fits
        "capability_coverage": float,  // How many required capabilities are covered
        "maturity_alignment": float,   // How well maturity levels align
        "semantic_relevance": float,   // Semantic similarity of descriptions
        "keyword_relevance": float,    // Relevance of matched keywords
        "context_alignment": float     // Overall context alignment
    }},
    "term_match_details": {{
        "matched_keywords": [],       // Technical keywords found in context
        "matched_capabilities": [],   // Capabilities demonstrated
        "matched_business_terms": [], // Business terms found in context
        "context_matches": [],        // Other relevant contextual matches
        "semantic_concepts": {{
            "technical_concepts": [
                {{
                    "concept": string,     // Technical concept identified
                    "relevance": float,    // Relevance score
                    "evidence": string     // Evidence from text
                }}
            ],
            "business_concepts": [
                {{
                    "concept": string,     // Business concept identified
                    "relevance": float,    // Relevance score
                    "evidence": string     // Evidence from text
                }}
            ]
        }}
    }},
    "improvement_notes": [],          // Notes for improving the match
    "suggestions": []                 // Suggestions for classification system
}}"""
            },
            {"role": "user", "content": f"""
EVALUATION CRITERIA:
1. Technical Alignment:
   - Does the use case implementation align with the category's core capabilities?
   - Are the required technical components present?
   - Do the outputs match expected category outcomes?

2. Business Alignment:
   - Does the use case purpose align with the category's business objectives?
   - Are the benefits consistent with category capabilities?
   - Is the maturity level appropriate?

3. Implementation Context:
   - Is the development approach consistent with the category?
   - Does the system architecture fit the category pattern?
   - Are there any technical conflicts or misalignments?

USE CASE DETAILS:
{use_case_text}

TECHNOLOGY CATEGORY DETAILS:
Name: {category['name']}
Definition: {category.get('definition', 'No definition available')}
Zone: {category.get('zone', 'Unknown')}
Maturity Level: {category.get('maturity_level', 'Unknown')}

Technical Keywords (Must be contextually relevant):
{', '.join(keywords)}

Core Capabilities (Should be demonstrated):
{', '.join(capabilities)}

Business/Domain Language (Should align):
{', '.join(business_terms)}

Current match_type: {match_type}
Initial confidence: {confidence}

Analyze this match and provide your assessment in the specified JSON format."""}
        ]
        
        # Get LLM analysis
        llm_result = await self._call_openai(messages)
        
        # Convert to AIClassification compatible format with enhanced scoring
        classification_data = {
            "match_type": llm_result["best_match"]["match_type"],
            "confidence": llm_result["best_match"]["confidence"],
            "analysis_method": "LLM",
            "analysis_version": "v2.2",
            "llm_verification": llm_result["agrees"],
            "llm_confidence": llm_result["best_match"]["confidence"],
            "llm_reasoning": llm_result["best_match"]["reasoning"],
            "field_match_scores": {
                **llm_result["field_analysis"],
                "semantic_relevance": llm_result["field_analysis"].get("semantic_relevance", 0.0),
                "keyword_relevance": llm_result["field_analysis"].get("keyword_relevance", 0.0),
                "context_alignment": llm_result["field_analysis"].get("context_alignment", 0.0)
            },
            "term_match_details": {
                **llm_result["matched_terms"],
                "semantic_concepts": llm_result["matched_terms"].get("semantic_concepts", {
                    "technical_concepts": [],
                    "business_concepts": []
                })
            },
            "matched_keywords": llm_result["matched_terms"]["technical_terms"],
            "llm_suggestions": llm_result["alternative_matches"],
            "improvement_notes": llm_result["improvement_notes"],
            "false_positive": not llm_result["agrees"] and confidence > 0.3,
            "manual_override": False,
            "review_status": "PENDING"
        }
        
        return classification_data

    async def analyze_no_match(self, use_case_text: str, current_category: Optional[str] = None) -> Dict[str, Any]:
        """Analyze why a use case doesn't match any categories."""
        try:
            if not use_case_text:
                return {
                    "reason": "Error: Empty use case text provided",
                    "reason_category": "UNCLEAR_DESC",
                    "confidence": 0.0,
                    "suggestions": []
                }

            # Build prompt with official categories and emerging technology guidance
            messages = [
                {"role": "system", "content": """You are an expert AI technology analyst evaluating federal use cases.
Your task is to:
1. Determine if the use case matches any of our official AI technology categories
2. If no match, analyze whether this represents an emerging technology pattern

Official AI Technology Categories:
{}

Provide your response in JSON format with the following structure:
{{
    "matches_existing": boolean,     // Whether it matches an existing category
    "closest_category": string,      // Name of closest official category if any
    "confidence": float,             // Confidence in the analysis (0.0-1.0)
    "reason": string,                // Primary reason for no match
    "reason_category": string,       // UNCLEAR_DESC, MISSING_INFO, EMERGING_TECH, TECH_MISMATCH, SCOPE_MISMATCH
    "emerging_pattern": {{           // Only if reason_category is EMERGING_TECH
        "pattern_name": string,      // Suggested name for new category
        "description": string,       // Description of the emerging pattern
        "similar_cases": string[],   // Examples of similar use cases
        "key_technologies": string[], // Core technologies involved
        "differentiation": string,   // How it differs from existing categories
        "market_evidence": string    // Evidence of this being an emerging trend
    }},
    "improvement_suggestions": {{
        "description": string[],     // Suggestions for description clarity
        "technical_detail": string[],// Technical details to add
        "scope": string[],          // Scope clarification needed
        "categorization": string[]   // Suggestions for better categorization
    }}
}}""".format('\n'.join(f"- {cat}" for cat in OFFICIAL_CATEGORIES))},
                {"role": "user", "content": f"""Analyze this use case and determine if it:
1. Matches any of our official categories (even partially)
2. Represents an emerging technology pattern
3. Needs more information for proper classification

Use Case:
{use_case_text}"""}
            ]

            response = await self.openai_client.post(
                "/chat/completions",
                json={
                    "model": self.settings.openai_model or "gpt-4",
                    "messages": messages,
                    "temperature": 0.7,
                    "response_format": { "type": "json_object" }
                }
            )
            response.raise_for_status()
            
            try:
                result = json.loads(response.json()["choices"][0]["message"]["content"])
                return {
                    "matches_existing": result.get("matches_existing", False),
                    "closest_category": result.get("closest_category"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "reason": result.get("reason", "Unknown reason"),
                    "reason_category": result.get("reason_category", "OTHER"),
                    "emerging_pattern": result.get("emerging_pattern") if result.get("reason_category") == "EMERGING_TECH" else None,
                    "improvement_suggestions": result.get("improvement_suggestions", {
                        "description": [],
                        "technical_detail": [],
                        "scope": [],
                        "categorization": []
                    })
                }
            except Exception as e:
                return {
                    "reason": f"Error: Failed to parse OpenAI response - {str(e)}",
                    "reason_category": "OTHER",
                    "confidence": 0.0,
                    "suggestions": []
                }
                
        except httpx.TimeoutException:
            return {
                "reason": "Error: Request to OpenAI timed out",
                "reason_category": "OTHER",
                "confidence": 0.0,
                "suggestions": []
            }
        except Exception as e:
            return {
                "reason": f"Error: {str(e)}",
                "reason_category": "OTHER",
                "confidence": 0.0,
                "suggestions": []
            }

    async def suggest_improvements(
        self,
        category_name: str,
        recent_matches: List[Dict],
        recent_failures: List[Dict]
    ) -> Dict:
        """
        Analyze classification patterns and suggest improvements to an AI technology category
        based on real-world usage data.
        """
        if category_name not in self.category_definitions:
            raise ValueError(f"Unknown category: {category_name}")
            
        category = self.category_definitions[category_name]
        
        # Safely get list properties with defaults
        keywords = category.get('keywords', [])
        if keywords is None:
            keywords = []
        elif isinstance(keywords, str):
            keywords = [keywords]
            
        capabilities = category.get('capabilities', [])
        if capabilities is None:
            capabilities = []
        elif isinstance(capabilities, str):
            capabilities = [capabilities]
            
        business_terms = category.get('business_language', [])
        if business_terms is None:
            business_terms = []
        elif isinstance(business_terms, str):
            business_terms = [business_terms]

        # Enhanced prompt for category improvement analysis
        messages = [
            {"role": "system", "content": """You are an expert AI technology analyst tasked with optimizing our AI technology classification framework.
Your goal is to analyze real-world classification patterns and suggest improvements to make this category more accurate and useful.

Provide your response in JSON format with the following structure:
{
    "definition_updates": {
        "current": string,           // Current category definition
        "suggested": string,         // Suggested updated definition
        "reasoning": string,         // Reasoning for changes
        "impact_analysis": {
            "clarity": string,       // Impact on definition clarity
            "coverage": string,      // Impact on use case coverage
            "precision": string      // Impact on classification precision
        }
    },
    "keyword_updates": {
        "add": string[],            // Keywords to add
        "remove": string[],         // Keywords to remove
        "modify": string[],         // Keywords to modify
        "reasoning": string         // Reasoning for changes
    },
    "capability_updates": {
        "add": string[],           // Capabilities to add
        "remove": string[],        // Capabilities to remove
        "modify": string[],        // Capabilities to modify
        "reasoning": string        // Reasoning for changes
    },
    "business_term_updates": {
        "add": string[],           // Business terms to add
        "remove": string[],        // Business terms to remove
        "context_improvements": string[], // Context improvements
        "reasoning": string        // Reasoning for changes
    },
    "match_criteria_updates": {
        "threshold_adjustments": {
            "confidence": float,    // Suggested confidence threshold
            "keyword_weight": float, // Suggested keyword weight
            "semantic_weight": float // Suggested semantic weight
        },
        "scoring_weights": {
            "technical": float,     // Technical alignment weight
            "business": float,      // Business alignment weight
            "context": float        // Context relevance weight
        },
        "reasoning": string,       // Reasoning for adjustments
        "expected_impact": {
            "false_positives": string, // Impact on false positives
            "false_negatives": string, // Impact on false negatives
            "overall_accuracy": string  // Impact on overall accuracy
        }
    }
}"""},
            {"role": "user", "content": f"""
CATEGORY DETAILS:
Name: {category.get('name', 'Unknown')}
Definition: {category.get('definition', 'No definition available')}
Technical Keywords: {', '.join(keywords)}
Core Capabilities: {', '.join(capabilities)}
Business Terms: {', '.join(business_terms)}
Zone: {category.get('zone', 'Unknown')}
Maturity Level: {category.get('maturity_level', 'Unknown')}

CLASSIFICATION HISTORY:
Successful Matches:
{json.dumps(recent_matches, indent=2)}

Failed Matches:
{json.dumps(recent_failures, indent=2)}

Analyze this category and suggest improvements in the specified JSON format."""}
        ]

        try:
            # Get LLM analysis
            response = await self._call_openai(messages)
            if not response or not isinstance(response, dict):
                logger.error("Invalid response from OpenAI")
                return self._get_default_improvement_response(category)

            # Ensure all required sections exist with proper structure
            result = {
                "definition_updates": {
                    "current": category.get('definition', 'No definition available'),
                    "suggested": response.get("definition_updates", {}).get("suggested", "No suggestions available"),
                    "reasoning": response.get("definition_updates", {}).get("reasoning", "No reasoning provided"),
                    "impact_analysis": response.get("definition_updates", {}).get("impact_analysis", {
                        "clarity": "No analysis available",
                        "coverage": "No analysis available",
                        "precision": "No analysis available"
                    })
                },
                "keyword_updates": response.get("keyword_updates", {
                    "add": [],
                    "remove": [],
                    "modify": [],
                    "reasoning": "No keyword analysis available"
                }),
                "capability_updates": response.get("capability_updates", {
                    "add": [],
                    "remove": [],
                    "modify": [],
                    "reasoning": "No capability analysis available"
                }),
                "business_term_updates": response.get("business_term_updates", {
                    "add": [],
                    "remove": [],
                    "context_improvements": [],
                    "reasoning": "No business term analysis available"
                }),
                "match_criteria_updates": response.get("match_criteria_updates", {
                    "threshold_adjustments": {
                        "confidence": 0.5,
                        "keyword_weight": 0.33,
                        "semantic_weight": 0.33
                    },
                    "scoring_weights": {
                        "technical": 0.4,
                        "business": 0.3,
                        "context": 0.3
                    },
                    "reasoning": "No criteria analysis available",
                    "expected_impact": {
                        "false_positives": "Unknown",
                        "false_negatives": "Unknown",
                        "overall_accuracy": "Unknown"
                    }
                })
            }

            return result

        except Exception as e:
            logger.error(f"Error in suggest_improvements: {str(e)}")
            return self._get_default_improvement_response(category)

    def _get_default_improvement_response(self, category: Dict) -> Dict:
        """Get a default response structure for improvements with category details."""
        return {
            "definition_updates": {
                "current": category.get('definition', 'No definition available'),
                "suggested": "No suggestions available",
                "reasoning": "Unable to analyze category",
                "impact_analysis": {
                    "clarity": "No analysis available",
                    "coverage": "No analysis available",
                    "precision": "No analysis available"
                }
            },
            "keyword_updates": {
                "add": [],
                "remove": [],
                "modify": [],
                "reasoning": "No keyword analysis available"
            },
            "capability_updates": {
                "add": [],
                "remove": [],
                "modify": [],
                "reasoning": "No capability analysis available"
            },
            "business_term_updates": {
                "add": [],
                "remove": [],
                "context_improvements": [],
                "reasoning": "No business term analysis available"
            },
            "match_criteria_updates": {
                "threshold_adjustments": {
                    "confidence": 0.5,
                    "keyword_weight": 0.33,
                    "semantic_weight": 0.33
                },
                "scoring_weights": {
                    "technical": 0.4,
                    "business": 0.3,
                    "context": 0.3
                },
                "reasoning": "No criteria analysis available",
                "expected_impact": {
                    "false_positives": "Unknown",
                    "false_negatives": "Unknown",
                    "overall_accuracy": "Unknown"
                }
            }
        } 