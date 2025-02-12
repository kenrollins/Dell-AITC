"""
Analysis models and enums for AI technology classification
"""

from enum import Enum, auto
from typing import Optional, List, Dict

class AnalysisMethod(str, Enum):
    """Analysis methods for classification"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    LLM = "llm"
    ALL = "all"

class MatchType(str, Enum):
    """Types of matches between use cases and AI technologies"""
    PRIMARY = "primary"      # Main solution recommendation
    SUPPORTING = "supporting"  # Additional technologies needed
    RELATED = "related"      # Potentially relevant technologies
    NO_MATCH = "no_match"    # No clear match found

class MatchResult:
    """Structured result for a technology match"""
    def __init__(
        self,
        category_name: str,
        match_type: MatchType,
        confidence: float,
        matched_keywords: List[str] = None,
        semantic_score: float = 0.0,
        keyword_score: float = 0.0,
        llm_score: float = 0.0,
        llm_explanation: str = "",
        match_method: str = "",
        match_details: Dict = None
    ):
        self.category_name = category_name
        self.match_type = match_type
        self.confidence = confidence
        self.matched_keywords = matched_keywords or []
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.llm_score = llm_score
        self.llm_explanation = llm_explanation
        self.match_method = match_method
        self.match_details = match_details or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/transmission"""
        return {
            'category_name': self.category_name,
            'match_type': self.match_type.value,
            'confidence': self.confidence,
            'matched_keywords': self.matched_keywords,
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'llm_score': self.llm_score,
            'llm_explanation': self.llm_explanation,
            'match_method': self.match_method,
            'match_details': self.match_details
        }

class AnalysisRequest:
    """Request parameters for classification"""
    def __init__(
        self,
        method: AnalysisMethod = AnalysisMethod.ALL,
        dry_run: bool = False,
        limit: Optional[int] = None,
        min_confidence: float = 0.25,
        require_primary: bool = True  # Whether to require at least one primary match
    ):
        self.method = method
        self.dry_run = dry_run
        self.limit = limit
        self.min_confidence = min_confidence
        self.require_primary = require_primary 