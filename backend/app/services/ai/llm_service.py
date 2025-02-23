"""
Dell-AITC LLM Service (v2.2)
Handles interactions with OpenAI's API for AI use case classification.
"""

import logging
from openai import AsyncOpenAI, APIError
from typing import Dict, Any, Optional
from ...config import get_settings
import json
import httpx

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM service with API key."""
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
        self.logger.info(f"Initializing OpenAI client with API key: {self.api_key[:8]}...")
        
        # Initialize OpenAI client with appropriate base URL for project-specific keys
        base_url = "https://api.projectaria.com/openai" if self.api_key.startswith('sk-proj-') else None
        self.logger.info(f"Using base URL: {base_url or 'default OpenAI'}")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries
        )
        
        # Enhanced prompt templates
        self.CLASSIFICATION_PROMPT = """
You are evaluating if a federal AI use case matches an AI technology category.
Consider the following schema-defined properties and relationships:

Category Properties:
{category_properties}
- Name: {name}
- Definition: {category_definition}
- Zone: {zone}
- Maturity Level: {maturity_level}
- Keywords: {keywords}
- Capabilities: {capabilities}
- Integration Patterns: {integration_patterns}
- Dependencies: {dependencies}

Use Case Details:
{use_case_properties}
- Name: {name}
- Description: {description}
- Purpose & Benefits: {purpose_benefits}
- Outputs: {outputs}
- Development Stage: {dev_stage}
- Implementation Method: {dev_method}

Evaluation Criteria:
1. Technical Alignment
   - Does the use case align with the category's core capabilities?
   - Is the maturity level appropriate?
   - Does the implementation match known integration patterns?

2. Keyword Analysis
   - Technical term matches: {technical_keywords}
   - Business language matches: {business_keywords}
   - Capability alignment: {capability_matches}

3. Zone Compatibility
   - Does the use case fit within the category's technical zone?
   - Are there cross-zone dependencies to consider?

4. Implementation Context
   - Development stage compatibility
   - Required capabilities present
   - Integration pattern feasibility

Provide your analysis in JSON format:
{
    "match_type": "PRIMARY" | "SUPPORTING" | "RELATED" | "NONE",
    "confidence": float,  // 0.0-1.0
    "field_match_scores": {
        "technical_alignment": float,
        "keyword_relevance": float,
        "capability_match": float,
        "zone_compatibility": float,
        "implementation_fit": float
    },
    "matched_terms": {
        "technical_terms": [string],
        "business_terms": [string],
        "capabilities": [string]
    },
    "justification": string,
    "improvement_suggestions": [string]
}
"""

    async def analyze_text(self, prompt: str) -> Dict[str, Any]:
        """Analyze text using OpenAI API with structured output."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an AI technology classifier analyzing federal use cases.
                    Provide your analysis in the following JSON format:
                    {
                        "match_type": "PRIMARY|SUPPORTING|RELATED|NO_MATCH",
                        "confidence": float between 0 and 1,
                        "justification": "Detailed explanation of the match",
                        "suggestions": ["List of improvement suggestions"],
                        "technical_terms": ["List of relevant technical terms found"],
                        "business_terms": ["List of relevant business terms found"]
                    }"""
                },
                {"role": "user", "content": str(prompt)}  # Ensure prompt is string
            ]
            
            settings = get_settings()
            self.logger.debug(f"Sending request to OpenAI API using model {settings.openai_model}...")
            
            try:
                self.logger.info(f"Making request to OpenAI API with model {settings.openai_model}")
                self.logger.info(f"Using base URL: {self.client.base_url}")
                
                response = await self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                # Parse and validate response
                result = json.loads(response.choices[0].message.content)
                
                # Ensure all fields are of correct type
                return {
                    "match_type": str(result.get("match_type", "NO_MATCH")),
                    "confidence": float(result.get("confidence", 0.0)),
                    "justification": str(result.get("justification", "No justification provided")),
                    "suggestions": [str(s) for s in result.get("suggestions", [])],
                    "technical_terms": [str(t) for t in result.get("technical_terms", [])],
                    "business_terms": [str(t) for t in result.get("business_terms", [])]
                }
                
            except httpx.ConnectError as e:
                self.logger.error(f"Connection error to OpenAI API: {str(e)}")
                self.logger.error(f"Base URL: {self.client.base_url}")
                self.logger.error(f"Request details: {e.request.url if hasattr(e, 'request') else 'No URL'}")
                raise
                
            except httpx.TimeoutException as e:
                self.logger.error(f"Timeout error: {str(e)}")
                raise
                
            except httpx.HTTPError as e:
                self.logger.error(f"HTTP error: {str(e)}")
                self.logger.error(f"Response status: {e.response.status_code if hasattr(e, 'response') else 'No status'}")
                self.logger.error(f"Response body: {e.response.text if hasattr(e, 'response') else 'No body'}")
                raise
            
        except APIError as e:
            self.logger.error(f"OpenAI API Error: {str(e)}")
            if "invalid_api_key" in str(e):
                self.logger.error(f"API Key validation failed. Key format: {self.api_key[:8]}...")
            return {
                "match_type": "ERROR",
                "confidence": 0.0,
                "justification": f"OpenAI API Error: {str(e)}",
                "suggestions": [],
                "technical_terms": [],
                "business_terms": []
            }
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}", exc_info=True)
            return {
                "match_type": "ERROR",
                "confidence": 0.0,
                "justification": f"Error in LLM analysis: {str(e)}",
                "suggestions": [],
                "technical_terms": [],
                "business_terms": []
            }

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self.client, 'close'):
            await self.client.close() 