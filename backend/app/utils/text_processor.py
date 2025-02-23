"""Text processing utilities for AI classification"""

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from ..config import get_settings

class TextProcessor:
    """Text processing utilities for embeddings and similarity"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize text processor
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the text processor"""
        self.logger.info(f"Initializing semantic model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
    async def cleanup(self):
        """Clean up resources"""
        pass  # No cleanup needed for sentence transformers
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Numpy array of embedding vector
        """
        if not text:
            return np.zeros(384)  # Default embedding size for MiniLM
            
        return self.model.encode(text, convert_to_numpy=True)
        
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
    def find_similar_texts(
        self,
        query: str,
        texts: List[str],
        min_similarity: float = 0.5,
        max_results: int = 5
    ) -> List[Dict[str, float]]:
        """Find most similar texts to a query
        
        Args:
            query: Query text to compare against
            texts: List of texts to search
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of dicts with text and similarity score
        """
        query_embedding = self.get_embedding(query)
        results = []
        
        for text in texts:
            text_embedding = self.get_embedding(text)
            similarity = self.calculate_similarity(query_embedding, text_embedding)
            
            if similarity >= min_similarity:
                results.append({
                    'text': text,
                    'similarity': similarity
                })
                
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]
        
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