"""
Configuration settings for the application.
Uses environment variables with pydantic for validation.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Data paths
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    input_dir: Path = data_dir / "input"
    output_dir: Path = data_dir / "output"
    
    # Model settings
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.30
    keyword_threshold: float = 0.25
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_timeout: float = 60.0  # Increased timeout
    openai_max_retries: int = 3
    openai_retry_delays: List[float] = [2.0, 4.0, 8.0]  # Exponential backoff
    
    # Sentence transformer settings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields in .env without raising errors

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
