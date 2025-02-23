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
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")  # Must be provided via env
    
    # Data paths
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    input_dir: Path = data_dir / "input"
    output_dir: Path = data_dir / "output"
    
    # Model settings
    semantic_model: str = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.30"))
    keyword_threshold: float = float(os.getenv("KEYWORD_THRESHOLD", "0.25"))
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "60.0"))
    openai_max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    openai_retry_delays: List[float] = [2.0, 4.0, 8.0]
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Sentence transformer settings
    sentence_transformer_model: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
        
        # Model aliases
        model_config = {
            "env_prefix": "",
            "env_file_encoding": "utf-8",
            "arbitrary_types_allowed": True
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings(_env_file='.env')
