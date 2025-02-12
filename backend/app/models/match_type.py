"""
Enum for AI technology classification match types.
"""

from enum import Enum, auto

class MatchType(str, Enum):
    """Types of matches between use cases and AI technologies"""
    PRIMARY = "PRIMARY"      # Main solution recommendation
    SUPPORTING = "SUPPORTING"  # Additional technologies needed
    RELATED = "RELATED"      # Potentially relevant technologies
    NO_MATCH = "NO_MATCH"    # No clear match found 