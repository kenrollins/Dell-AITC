import os
from typing import Dict, Any, Optional
import json
import logging
from openai import AsyncOpenAI
from ..config import get_settings

class LLMService:
    def __init__(self):
        """Initialize the LLM service with API key from settings."""
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model or "gpt-4-turbo-preview"
        self.logger = logging.getLogger(__name__)
        
    async def analyze_text(self, prompt: str) -> Dict[str, Any]:
        """Analyze text using OpenAI API with structured output.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dict containing the structured analysis
        """
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
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI API Error: {str(e)}")
            return {
                "match_type": "NO_MATCH",
                "confidence": 0.0,
                "justification": f"Error during analysis: {str(e)}",
                "suggestions": [],
                "technical_terms": [],
                "business_terms": []
            } 