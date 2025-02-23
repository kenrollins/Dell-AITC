import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities for embeddings and similarity"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize text processor
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    async def initialize(self):
        """Initialize the text processor by loading the model"""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")
        
    def cleanup(self):
        """Clean up resources"""
        # No cleanup needed currently
        pass
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Numpy array of embedding
        """
        if not text:
            # Return zero vector for empty text
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
            
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
            
        # Get embedding from model
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.embeddings_cache[text] = embedding
        return embedding
        
    def calculate_similarity(self, text1: Union[str, np.ndarray], text2: Union[str, np.ndarray]) -> float:
        """Calculate cosine similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Handle string inputs
        if isinstance(text1, str) and isinstance(text2, str):
            if not text1 or not text2:
                return 0.0
            text1 = self.get_embedding(text1)
            text2 = self.get_embedding(text2)
        
        # Handle numpy arrays
        if isinstance(text1, np.ndarray) and isinstance(text2, np.ndarray):
            if text1 is None or text2 is None or text1.size == 0 or text2.size == 0:
                return 0.0
            
            # Ensure vectors are 2D for batch processing
            if len(text1.shape) == 1:
                text1 = text1.reshape(1, -1)
            if len(text2.shape) == 1:
                text2 = text2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(text1, text2)
            return float(similarity[0][0])  # Return scalar value
        
        raise ValueError("Inputs must be either both strings or both numpy arrays")
        
    def find_similar_texts(self, query: str, texts: List[str], min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Find texts similar to query
        
        Args:
            query: Query text to find similar texts for
            texts: List of texts to search in
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (text, similarity) tuples sorted by similarity
        """
        if not query or not texts:
            return []
            
        # Get query embedding
        query_emb = self.get_embedding(query)
        
        # Get embeddings for all texts
        text_embs = np.stack([self.get_embedding(text) for text in texts])
        
        # Calculate similarities
        similarities = util.cos_sim(query_emb, text_embs)[0]
        
        # Get texts above threshold
        results = []
        for text, sim in zip(texts, similarities):
            if sim >= min_similarity:
                results.append((text, float(sim)))
                
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results 