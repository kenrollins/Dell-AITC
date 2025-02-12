"""
Enum for AI technology classification match types.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class MatchType(Enum):
    """Types of matches between use cases and AI technologies"""
    PRIMARY = "PRIMARY"      # Main solution recommendation
    SUPPORTING = "SUPPORTING"  # Additional technologies needed
    RELATED = "RELATED"      # Potentially relevant technologies
    NO_MATCH = "NO_MATCH"    # No clear match found 

@dataclass
class MatchResult:
    """Structured result for AI technology category matching"""
    use_case_id: str
    primary_matches: List[Dict[str, Any]]
    supporting_matches: List[Dict[str, Any]]
    related_matches: List[Dict[str, Any]]
    match_method: str
    confidence: float
    field_match_scores: Dict[str, float]
    matched_terms: Dict[str, List[str]]
    llm_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def has_matches(self) -> bool:
        """Check if any matches were found"""
        return bool(self.primary_matches or self.supporting_matches or self.related_matches)

    def get_best_match(self) -> Optional[Dict[str, Any]]:
        """Get the highest confidence match across all types"""
        all_matches = (
            [(match, "PRIMARY") for match in self.primary_matches] +
            [(match, "SUPPORTING") for match in self.supporting_matches] +
            [(match, "RELATED") for match in self.related_matches]
        )
        if not all_matches:
            return None
        best_match, match_type = max(all_matches, key=lambda x: x[0]["confidence"])
        return {**best_match, "match_type": match_type}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the result, similar to dictionary get method"""
        if hasattr(self, key):
            return getattr(self, key)
        return default 