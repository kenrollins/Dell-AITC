"""Text processing utilities for AI classification"""

import re
import logging
import numpy as np
from typing import Dict, Optional, Union
from sentence_transformers import SentenceTransformer, util
from ..config import get_settings

class TextProcessor:
    """Handle text processing and embedding operations"""
    
    DEFAULT_MODEL = 'all-mpnet-base-v2'  # Better performance than MiniLM
    FALLBACK_MODEL = 'all-MiniLM-L6-v2'  # Lightweight fallback
    
    def __init__(self, model_name: str = None):
        try:
            settings = get_settings()
            self.model = SentenceTransformer(model_name or settings.sentence_transformer_model)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Initialized semantic model: {model_name or settings.sentence_transformer_model}")
        except Exception as e:
            self.logger.warning(f"Failed to load primary model, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def cleanup_text(self, text: Union[str, float, int, None], preserve_case: bool = False) -> str:
        """Standardize text for better matching"""
        if text is None:
            return ""
            
        # Convert any type to string
        text = str(text)
            
        # Convert to lowercase unless preserve_case is True
        if not preserve_case:
            text = text.lower()
            
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common abbreviations and contractions
        text = text.replace("ai/ml", "ai ml")
        text = text.replace("nlp/nlu", "nlp nlu")
        
        return text.strip()
        
    def get_embedding(self, text: Union[str, float, int, None], normalize: bool = True) -> np.ndarray:
        """Get embedding vector for text with improved preprocessing"""
        # Clean and standardize text
        text = self.cleanup_text(text)
        
        # Handle empty or invalid text
        if not text:
            return self.model.encode("", convert_to_tensor=True, normalize_embeddings=normalize)
            
        return self.model.encode(text, convert_to_tensor=True, normalize_embeddings=normalize)
        
    def calculate_semantic_similarity(self, 
                                   vec1: np.ndarray,
                                   vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors with validation"""
        try:
            similarity = float(util.pytorch_cos_sim(vec1, vec2)[0][0])
            return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
            
    def calculate_weighted_semantic_score(self,
                                       use_case_text: Dict[str, Union[str, float, int, None]],
                                       category_text: Dict[str, str],
                                       field_weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate semantic similarity with field weighting"""
        if not field_weights:
            field_weights = {
                'description': 0.4,
                'purpose_benefits': 0.3,
                'outputs': 0.2,
                'systems': 0.1
            }
            
        total_score = 0.0
        total_weight = 0.0
        
        for field, weight in field_weights.items():
            if field in use_case_text and field in category_text:
                # Safely convert both texts to strings
                use_case_field = self.cleanup_text(use_case_text[field])
                category_field = self.cleanup_text(category_text[field])
                
                if use_case_field and category_field:  # Only process if both have content
                    vec1 = self.get_embedding(use_case_field)
                    vec2 = self.get_embedding(category_field)
                    score = self.calculate_semantic_similarity(vec1, vec2)
                    total_score += score * weight
                    total_weight += weight
                
        if total_weight > 0:
            return round(total_score / total_weight, 3)
        return 0.0 