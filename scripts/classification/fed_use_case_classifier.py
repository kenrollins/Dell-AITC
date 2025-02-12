#!/usr/bin/env python3
"""
Federal Use Case AI Technology Classifier

This script evaluates and classifies federal AI use cases against a standardized set of 
AI technology categories using a multi-method approach that combines:
- Keyword-based matching with relevance scoring
- Semantic similarity analysis using sentence transformers
- LLM-based verification (Ollama locally or OpenAI)

Key Features:
- Direct Neo4j integration for data persistence
- Multi-method classification with weighted scoring
- Sophisticated confidence scoring with method agreement boosting
- Detailed analysis of unmatched cases
- Support for both local (Ollama) and cloud (OpenAI) LLM processing
- Dry-run capability for testing
- Batch processing with progress tracking
- Comprehensive logging and statistics

Usage Examples:
    # Process all cases using local Ollama (testing)
    python fed_use_case_classifier.py -a --llm-provider ollama

    # Process 10 cases in dry-run mode
    python fed_use_case_classifier.py -n 10 --dry-run

    # Production run using OpenAI
    python fed_use_case_classifier.py -a --llm-provider openai

Arguments:
    -n, --number NUM    Number of use cases to process
    -a, --all          Process all unclassified use cases
    --dry-run          Run without making database changes
    --llm-provider     LLM provider to use (ollama or openai, default: openai)
    --batch-size       Number of cases per batch (default: 10)
    --compare-deprecated Compare results with deprecated classifier
    --verify-env       Verify environment setup

Environment Variables:
    NEO4J_URI         Neo4j database URI
    NEO4J_USER        Neo4j username
    NEO4J_PASSWORD    Neo4j password
    OPENAI_API_KEY    OpenAI API key (required if using openai provider)
    OLLAMA_BASE_URL   Ollama base URL (defaults to http://localhost:11434)

Output:
    1. Creates AIClassification relationships in Neo4j with:
       - Classification type (PRIMARY, SECONDARY, RELATED)
       - Confidence scores from each method
       - Detailed justification and analysis
    
    2. Creates UnmatchedAnalysis nodes for cases without matches:
       - Reason categorization
       - Detailed LLM analysis
       - Improvement suggestions
       - Potential new category recommendations

    3. Generates detailed logs with:
       - Processing statistics
       - Method performance metrics
       - Error tracking
       - Batch processing progress

Author: Dell Federal Professional Services
Version: 2.0.0
"""

import os
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum, auto
from neo4j import GraphDatabase
from neo4j.graph import Node
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import Tensor
from sentence_transformers import util
import re

# =============================================================================
# Configuration
# =============================================================================

# After imports, before Config class
# Configure logging with force=True to override any existing handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Ensure stdout is flushed immediately
sys.stdout.reconfigure(line_buffering=True)

class MatchMethod(Enum):
    """Enum for classification methods"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    LLM = "llm"

class RelationType(Enum):
    """Enum for relationship types"""
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    RELATED = "RELATED"
    NO_MATCH = "NO_MATCH"

def load_neo4j_schema() -> Dict[str, Any]:
    """Load Neo4j schema from documentation"""
    # rule: schema_changes: Use master schema from docs/neo4j directory
    schema_path = Path("docs/neo4j/neo4j_schema.json")
    
    if not schema_path.exists():
        logging.error(f"❌ Master schema not found at {schema_path}")
        logging.error("Please ensure the schema file exists in the docs/neo4j directory")
        return {}
    
    try:
        with open(schema_path) as f:
            schema = json.load(f)
            
        # Validate schema version
        if 'version' not in schema:
            logging.error("❌ Schema missing version information")
            return {}
            
        logging.info(f"✓ Loaded Neo4j schema version {schema['version']}")
        
        # Validate required sections
        required_sections = ['nodes', 'relationships']
        for section in required_sections:
            if section not in schema:
                logging.error(f"❌ Schema missing required section: {section}")
                return {}
            else:
                logging.info(f"✓ Found {len(schema[section])} {section}")
        
        # Validate core node types
        required_nodes = ['AICategory', 'UseCase', 'Agency', 'Bureau']
        for node in required_nodes:
            if node not in schema['nodes']:
                logging.error(f"❌ Schema missing required node type: {node}")
                return {}
            else:
                logging.info(f"✓ Validated {node} node schema")
        
        return schema
        
    except Exception as e:
        logging.error(f"❌ Error loading schema: {str(e)}")
        return {}

class Config:
    """Global configuration settings"""
    
    # Version for environment checks
    VERSION = "2.0.0"
    REQUIRED_ENV_VARS = {
        'base': ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'OPENAI_API_KEY']
    }
    
    # Load schema information
    SCHEMA = load_neo4j_schema()
    
    class LLM:
        """LLM settings"""
        MODEL = "gpt-4-turbo-preview"
        TEMPERATURE = 0.3
        MAX_TOKENS = 500
        TIMEOUT = 30
        
        # Keep existing prompts
        CLASSIFICATION_PROMPT = """
        You are evaluating if a federal AI use case matches a specific AI technology category.
        You should consider the following schema-defined relationships and properties:

        Category Properties:
        - category_definition: Detailed description of the technology
        - maturity_level: Technology maturity (Emerging, Established, etc.)
        - zone: Technical zone classification
        - capabilities: Core technical capabilities
        - integration_patterns: How the technology typically integrates

        Use Case Properties:
        - purpose_benefits: Intended benefits and objectives
        - outputs: Expected deliverables and results
        - dev_stage: Development stage
        - infrastructure: Technical infrastructure details
        - systems: Associated technical systems

        Analyze this use case:
        {use_case_text}

        Against this technology category:
        {category_text}

        Consider:
        1. Direct technology alignment with category definition
        2. Capability matching with required technical capabilities
        3. Integration pattern compatibility
        4. Maturity level appropriateness
        5. Zone alignment with use case infrastructure

        Provide your response in JSON format with the following fields:
        {
            "confidence": float,  # 0-1 score of match confidence
            "relationship_type": "PRIMARY" | "SECONDARY" | "RELATED" | "NO_MATCH",
            "justification": string,  # Detailed explanation using schema properties
            "key_alignments": [string],  # List of main points of alignment
            "potential_gaps": [string],  # List of any misalignments or gaps
            "capability_matches": [string],  # List of matching capabilities
            "integration_notes": string  # Notes about integration compatibility
        }
        """
        
        NO_MATCH_ANALYSIS_PROMPT = """
        Analyze why this federal AI use case doesn't match our existing AI technology categories.
        Consider our schema-defined technology classification system:

        Technology Zones:
        {zones}

        Core Capabilities:
        {capabilities}

        Integration Patterns:
        {patterns}

        Use Case:
        {use_case_text}

        Analyze and provide your response in JSON format with the following fields:
        {
            "reason_category": "NOVEL_TECH" | "IMPLEMENTATION_SPECIFIC" | "NON_AI" | "UNCLEAR_DESC" | "OTHER",
            "detailed_analysis": string,  # In-depth analysis of the use case
            "improvement_suggestions": [string],  # List of suggestions to improve categorization
            "potential_new_categories": [string],  # Suggestions for new technology categories
            "key_technologies": [string],  # List of key technologies mentioned
            "implementation_notes": string,  # Notes about the implementation approach
            "closest_zone": string,  # Most relevant existing technology zone
            "missing_capabilities": [string]  # Capabilities needed but not in current schema
        }
        """
    
    class Scoring:
        """Scoring and threshold settings"""
        # Method Weights
        WEIGHTS = {
            'keyword': {
                'score_weight': 0.3,
                'threshold': 0.4
            },
            'semantic': {
                'score_weight': 0.4,
                'threshold': 0.5
            },
            'llm': {
                'score_weight': 0.3,
                'threshold': 0.6
            }
        }
        
        # Thresholds for relationship types
        THRESHOLDS = {
            'primary': 0.8,
            'secondary': 0.6,
            'related': 0.4
        }
        
        # Boost factors
        BOOST = {
            'three_method': 0.30,    # All methods agree
            'semantic_llm': 0.20,    # Semantic + LLM agree
            'strong_two': 0.15,      # Any two methods with strong agreement
            'keyword_semantic': 0.10, # Keyword + Semantic agree
            'keyword_llm': 0.08,     # Keyword + LLM agree
            'strong_individual': 0.05 # Single method very strong
        }
        
        # Add schema-aware scoring adjustments
        ZONE_MATCH_BOOST = 0.05  # Boost for matching technology zone
        CAPABILITY_MATCH_BOOST = 0.02  # Boost per matching capability
        INTEGRATION_MATCH_BOOST = 0.03  # Boost for matching integration pattern
    
    class Processing:
        """Processing settings"""
        DEFAULT_BATCH_SIZE = 10
        MAX_RETRIES = 3
        TIMEOUT_SECONDS = 30
        CACHE_SIZE = 1000  # Number of embeddings to cache
        
        # Retry settings
        RETRY_DELAYS = [1, 5, 15]  # Seconds between retries
        
        # Memory thresholds
        MAX_MEMORY_PERCENT = 85.0  # Pause processing if memory usage exceeds this
        
        # Progress tracking
        PROGRESS_UPDATE_INTERVAL = 5  # Seconds between progress updates

# Core Data Classes
@dataclass
class Keyword:
    id: str
    name: str
    relevance_score: float
    source: str

@dataclass
class AICategory:
    id: str
    name: str
    category_definition: str
    status: str
    maturity_level: str
    zone: str
    created_at: datetime
    last_updated: datetime
    keywords: List[Keyword]
    capabilities: List[str]

@dataclass
class MatchResult:
    """Enhanced match result with detailed scoring"""
    use_case_id: str
    category_id: str
    method: Optional[MatchMethod] = None
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    llm_score: float = 0.0
    final_score: float = 0.0
    confidence: float = 0.0
    relationship_type: RelationType = RelationType.NO_MATCH
    matched_keywords: Optional[List[str]] = None
    semantic_details: Optional[str] = None
    llm_justification: Optional[str] = None
    key_alignments: Optional[List[str]] = None
    potential_gaps: Optional[List[str]] = None
    explanation: str = ""
    error: Optional[str] = None

    def calculate_final_score(self) -> float:
        """Calculate final score with enhanced method agreement boosting"""
        weights = Config.Scoring.WEIGHTS
        boost = Config.Scoring.BOOST
        
        # Calculate base weighted scores
        keyword_weighted = self.keyword_score * weights['keyword']['score_weight']
        semantic_weighted = self.semantic_score * weights['semantic']['score_weight']
        llm_weighted = self.llm_score * weights['llm']['score_weight']
        
        # Calculate boost based on method agreement
        boost_value = 0.0
        
        # Strong boost for three-method agreement
        if (self.keyword_score >= weights['keyword']['threshold'] and 
            self.semantic_score >= weights['semantic']['threshold'] and
            self.llm_score >= weights['llm']['threshold']):
            boost_value += boost['three_method']
            
        # Medium boost for semantic + LLM agreement
        elif (self.semantic_score >= weights['semantic']['threshold'] and 
              self.llm_score >= weights['llm']['threshold']):
            boost_value += boost['semantic_llm']
            
        # Medium-high boost for strong two-method agreement
        elif ((self.keyword_score >= 0.6 and self.semantic_score >= 0.6) or
              (self.keyword_score >= 0.6 and self.llm_score >= 0.6) or
              (self.semantic_score >= 0.6 and self.llm_score >= 0.6)):
            boost_value += boost['strong_two']
            
        # Additional boosts for strong individual signals
        if self.keyword_score >= 0.8:
            boost_value += boost['strong_individual']
        if self.semantic_score >= 0.7:
            boost_value += boost['strong_individual']
        if self.llm_score >= 0.8:
            boost_value += boost['strong_individual']
        
        # Calculate final score
        base_score = max(keyword_weighted, semantic_weighted, llm_weighted)
        final_score = min(1.0, base_score + boost_value)
        
        return round(final_score, 3)

    def determine_relationship_type(self) -> RelationType:
        """Determine relationship type with cross-validation"""
        thresholds = Config.Scoring.THRESHOLDS
        score = self.calculate_final_score()
        
        # Count methods that exceed their thresholds
        method_agreement = sum(1 for s in [
            (self.keyword_score, Config.Scoring.WEIGHTS['keyword']['threshold']),
            (self.semantic_score, Config.Scoring.WEIGHTS['semantic']['threshold']),
            (self.llm_score, Config.Scoring.WEIGHTS['llm']['threshold'])
        ] if s[0] >= s[1])
        
        # Adjust thresholds based on method agreement
        threshold_modifier = 0.95 if method_agreement >= 2 else 1.05
        
        if score >= thresholds['primary'] * threshold_modifier:
            return RelationType.PRIMARY
        elif score >= thresholds['secondary'] * threshold_modifier:
            return RelationType.SECONDARY
        elif score >= thresholds['related'] * threshold_modifier:
            return RelationType.RELATED
        
        return RelationType.NO_MATCH

@dataclass
class UnmatchedAnalysis:
    """Enhanced unmatched analysis with detailed information"""
    id: str
    timestamp: datetime
    use_case_id: str
    reason_category: str  # NOVEL_TECH, IMPLEMENTATION_SPECIFIC, NON_AI, UNCLEAR_DESC, OTHER
    llm_analysis: str    # Full LLM analysis text
    improvement_suggestions: List[str]
    potential_categories: List[str]
    key_technologies: List[str]
    implementation_notes: str
    best_scores: Dict[str, float]  # Scores from each method
    analyzed_at: datetime
    analyzed_by: str
    status: str  # NEW, REVIEWED, ACTIONED

class Neo4jClassifier:
    """Main classifier class with direct Neo4j integration"""
    
    def __init__(self, driver=None, dry_run=False):
        """Initialize the classifier with Neo4j driver"""
        if driver:
            self.driver = driver
        else:
            self.setup_neo4j_connection()
        self.setup_embeddings_model()
        self.setup_llm_client()
        self.categories = self.get_technology_categories()
        logging.info(f"Found {len(self.categories)} technology categories")
        self.text_processor = TextProcessor()
        self.llm_processor = LLMProcessor()
        self.schema_metadata = {
            'zones': [],
            'capabilities': [],
            'patterns': []
        }
        self.load_schema_metadata()
        self.dry_run = dry_run
        
    def setup_neo4j_connection(self):
        """Initialize Neo4j connection and verify schema"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, password]):
            raise ValueError("Missing required Neo4j environment variables")
            
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_schema()
        
    def _verify_schema(self):
        """Verify required Neo4j schema elements exist"""
        with self.driver.session() as session:
            # Check for required nodes
            for node_type in ["UseCase", "AICategory", "Zone", "Keyword", "Capability"]:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                count = result.single()["count"]
                logging.info(f"Found {count} nodes of type {node_type}")
                if count == 0:
                    logging.warning(f"No nodes found of type {node_type}")
            
            # Enhanced keyword type validation
            result = session.run("""
                MATCH (k:Keyword)
                RETURN k.type as type, count(k) as count
                ORDER BY count DESC
            """)
            logging.info("\nKeyword Type Distribution:")
            for record in result:
                logging.info(f"- {record['type']}: {record['count']} keywords")
            
            # Verify keyword types
            result = session.run("""
                MATCH (k:Keyword)
                WHERE NOT k.type IN ['technical_keywords', 'capabilities', 'business_language']
                RETURN count(k) as invalid_count
            """)
            invalid_count = result.single()["invalid_count"]
            if invalid_count > 0:
                logging.warning(f"Found {invalid_count} keywords with invalid type")

            # Check keyword relevance scores
            result = session.run("""
                MATCH (k:Keyword)
                WHERE k.relevance_score IS NULL
                RETURN count(k) as missing_score_count
            """)
            missing_scores = result.single()["missing_score_count"]
            if missing_scores > 0:
                logging.warning(f"Found {missing_scores} keywords missing relevance scores")

    def load_schema_metadata(self):
        """Load and cache schema metadata for matching"""
        # Initialize empty metadata
        self.schema_metadata = {
            'zones': [],
            'capabilities': [],
            'patterns': []
        }
        
        # Try to load each type of metadata separately
        with self.driver.session() as session:
            try:
                # Load zones if they exist
                result = session.run("MATCH (z:Zone) RETURN collect(z.name) as zones")
                zones = result.single()
                if zones and zones.get('zones'):
                    self.schema_metadata['zones'] = zones['zones']
                    
                # Load capabilities if they exist
                result = session.run("MATCH (c:Capability) RETURN collect(c.name) as capabilities")
                capabilities = result.single()
                if capabilities and capabilities.get('capabilities'):
                    self.schema_metadata['capabilities'] = capabilities['capabilities']
                    
                # Load integration patterns if they exist
                result = session.run("MATCH (p:IntegrationPattern) RETURN collect(p.name) as patterns")
                patterns = result.single()
                if patterns and patterns.get('patterns'):
                    self.schema_metadata['patterns'] = patterns['patterns']
                    
                logging.info(f"Loaded schema metadata: {len(self.schema_metadata['zones'])} zones, " + 
                           f"{len(self.schema_metadata['capabilities'])} capabilities, " +
                           f"{len(self.schema_metadata['patterns'])} patterns")
                           
            except Exception as e:
                logging.warning(f"Error loading schema metadata: {str(e)}")
                # Continue with empty metadata

    def setup_embeddings_model(self):
        """Initialize the sentence transformer model for embeddings"""
        self.text_processor = TextProcessor()
        logging.info("✓ Initialized sentence transformer model")
    
    def setup_llm_client(self):
        """Initialize the LLM client"""
        self.llm_processor = LLMProcessor()
        logging.info("✓ Initialized LLM processor")
    
    def _calculate_keyword_score(self, use_case_text: str, category: AICategory) -> float:
        """Calculate keyword-based score."""
        try:
            # Get category keywords
            keywords = category.keywords
            if not keywords:
                return 0.0
            
            # Convert to lowercase for case-insensitive matching
            use_case_text = use_case_text.lower()
            
            # Extract keyword names and convert to lowercase
            keyword_names = [k.lower() for k in keywords if k]
            if not keyword_names:
                return 0.0
            
            # Count matches
            matches = sum(1 for k in keyword_names if k in use_case_text)
            
            # Calculate score based on matches
            score = matches / len(keyword_names)
            
            # Normalize score to 0-1 range
            return min(1.0, score)
            
        except Exception as e:
            logging.error(f"Error calculating keyword score: {str(e)}")
            return 0.0

    def _calculate_semantic_score(self, use_case_text: str, category_text: str) -> float:
        """Calculate semantic similarity score."""
        try:
            # Get embeddings
            use_case_embedding = self.text_processor.get_embedding(use_case_text)
            category_embedding = self.text_processor.get_embedding(category_text)
            
            # Calculate cosine similarity
            similarity = self.text_processor.calculate_semantic_similarity(
                use_case_embedding,
                category_embedding
            )
            
            return similarity
            
        except Exception as e:
            logging.error(f"Error calculating semantic score: {str(e)}")
            return 0.0

    def _calculate_llm_score(self, use_case_text: str, category_text: str) -> Tuple[float, str]:
        """Calculate LLM-based verification score."""
        try:
            # Get LLM analysis
            response = self.llm_processor.verify_match(
                use_case_text=use_case_text,
                category_text=category_text
            )
            
            # Parse response
            if isinstance(response, dict):
                score = float(response.get('confidence', 0.0))
                analysis = response.get('justification', '')
            else:
                score = 0.0
                analysis = "Error: Invalid LLM response format"
            
            return score, analysis
            
        except Exception as e:
            logging.error(f"Error calculating LLM score: {str(e)}")
            return 0.0, f"Error during LLM verification: {str(e)}"
        
    def _determine_final_score(self, 
                             keyword_score: float,
                             semantic_score: float,
                             llm_score: float,
                             matched_keywords: Set[str]) -> Tuple[float, str]:
        """Determine final confidence score and relationship type"""
        # Base weights
        weights = {
            'keyword': 0.3,
            'semantic': 0.3,
            'llm': 0.4
        }
        
        # Adjust weights based on matched keywords
        if len(matched_keywords) >= 3:
            weights['keyword'] += 0.1
            weights['semantic'] -= 0.05
            weights['llm'] -= 0.05
        
        # Calculate weighted score
        final_score = (
            keyword_score * weights['keyword'] +
            semantic_score * weights['semantic'] +
            llm_score * weights['llm']
        )
        
        # Determine relationship type based on score thresholds
        if final_score >= 0.8:
            relationship_type = "PRIMARY"
        elif final_score >= 0.6:
            relationship_type = "SECONDARY"
        elif final_score >= 0.4:
            relationship_type = "RELATED"
        else:
            relationship_type = "NO_MATCH"
            
        return round(final_score, 3), relationship_type
        
    def _validate_use_case(self, use_case: dict) -> Tuple[bool, str]:
        """Validate that a use case has sufficient content for classification.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason if invalid)
        """
        # Check for ID (always required)
        if not use_case.get('id'):
            return False, "Use case ID is missing"
            
        # Check for at least one of description or purpose_benefits
        has_description = bool(use_case.get('description', '').strip())
        has_purpose = bool(use_case.get('purpose_benefits', '').strip())
        
        if not (has_description or has_purpose):
            return False, "Use case must have either description or purpose and benefits"
            
        # Get all available content
        content_fields = {
            'description': use_case.get('description', ''),
            'purpose_benefits': use_case.get('purpose_benefits', ''),
            'outputs': use_case.get('outputs', '')
        }
        
        # Check total content length (at least 20 chars for meaningful analysis)
        total_content = ' '.join(value for value in content_fields.values() if value)
        if len(total_content.strip()) < 20:
            return False, "Insufficient content for classification (minimum 20 characters required)"
        
        # Check for non-meaningful content
        low_value_patterns = [
            r'^n/?a$', r'^none$', r'^unknown$', r'^tbd$', r'^pending$',
            r'^not specified$', r'^not applicable$', r'^to be determined$'
        ]
        
        # Only check fields that are present
        for field, content in content_fields.items():
            if content and any(re.match(pattern, content.strip().lower()) for pattern in low_value_patterns):
                return False, f"Non-meaningful content in {field}"
            
        return True, ""

    def _prepare_use_case_text(self, use_case):
        """Prepare use case text for analysis by combining and weighting fields."""
        text_parts = []
        
        # Safely get text fields with fallback to empty string
        purpose_benefits = use_case.get('purpose_benefits', '') or ''
        outputs = use_case.get('outputs', '') or ''
        
        # Clean and weight the text fields
        if purpose_benefits.strip():
            # Repeat text based on weight (0.6)
            text_parts.extend([purpose_benefits.strip()] * 3)
        
        if outputs.strip():
            # Repeat text based on weight (0.4)
            text_parts.extend([outputs.strip()] * 2)
        
        if not text_parts:
            raise ValueError(f"Use case {use_case.get('name', use_case.get('id', 'Unknown'))} has no valid content for analysis")
        
        # Join with clear field separation
        return " [FIELD_BREAK] ".join(text_parts)

    def _prepare_category_text(self, category):
        """Prepare category text for analysis."""
        text_parts = []
        
        # Safely get text fields with fallback to empty string
        name = category.get('name', '') or ''
        description = category.get('description', '') or ''
        
        if name.strip():
            text_parts.append(name.strip())
        
        if description.strip():
            text_parts.append(description.strip())
        
        if not text_parts:
            raise ValueError(f"Category {category.get('id', 'Unknown')} has no valid content for analysis")
        
        return " [FIELD_BREAK] ".join(text_parts)

    def classify_use_case(self, use_case):
        """Classify a use case against available technology categories."""
        try:
            use_case_text = self._prepare_use_case_text(use_case)
            matches = []
            
            # First pass: Quick filtering using keyword and semantic matching
            candidates = []
            for category in self.categories:
                category_text = self._prepare_category_text(category)
                
                # Calculate keyword and semantic scores
                keyword_score = self._calculate_keyword_score(use_case_text, category)
                semantic_score = self._calculate_semantic_score(use_case_text, category_text)
                
                # Only proceed if either score is above threshold
                if keyword_score >= Config.Scoring.WEIGHTS['keyword']['threshold'] or \
                   semantic_score >= Config.Scoring.WEIGHTS['semantic']['threshold']:
                    candidates.append({
                        'category': category,
                        'category_text': category_text,
                        'keyword_score': keyword_score,
                        'semantic_score': semantic_score
                    })
            
            # Sort candidates by combined score
            top_candidates = sorted(
                candidates,
                key=lambda x: (
                    x['keyword_score'] * Config.Scoring.WEIGHTS['keyword']['score_weight'] +
                    x['semantic_score'] * Config.Scoring.WEIGHTS['semantic']['score_weight']
                ),
                reverse=True
            )
            
            # Second pass: Detailed LLM analysis only for promising candidates
            for candidate in top_candidates[:5]:  # Limit to top 5 candidates
                category = candidate['category']
                llm_score, llm_analysis = self._calculate_llm_score(
                    use_case_text,
                    candidate['category_text']
                )
                
                # Calculate final score
                final_score = (
                    candidate['keyword_score'] * Config.Scoring.WEIGHTS['keyword']['score_weight'] +
                    candidate['semantic_score'] * Config.Scoring.WEIGHTS['semantic']['score_weight'] +
                    llm_score * Config.Scoring.WEIGHTS['llm']['score_weight']
                )
                
                if final_score >= Config.Scoring.THRESHOLDS['related']:
                    relationship_type = self._determine_relationship_type(final_score)
                    match_result = {
                        'category_id': category['id'],
                        'category_name': category['name'],
                        'scores': {
                            'keyword_score': candidate['keyword_score'],
                            'semantic_score': candidate['semantic_score'],
                            'llm_score': llm_score,
                            'final_score': final_score
                        },
                        'relationship_type': relationship_type,
                        'llm_analysis': llm_analysis
                    }
                    matches.append(match_result)
            
            # Sort matches by final score
            if matches:
                matches.sort(key=lambda x: x['scores']['final_score'], reverse=True)
                relationship_type = matches[0]['relationship_type']
            else:
                relationship_type = 'NO_MATCH'
            
            result = {
                'use_case_id': use_case.get('id'),
                'matches': matches,
                'relationship_type': relationship_type,
                'scores': {  # Initialize empty scores for consistency
                    'keyword_score': 0.0,
                    'semantic_score': 0.0,
                    'llm_score': 0.0,
                    'final_score': 0.0
                }
            }
            
            # Update scores with best match if available
            if matches:
                result['scores'] = matches[0]['scores']
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing use case {use_case.get('id')}: {str(e)}")
            return {
                'use_case_id': use_case.get('id'),
                'matches': [],
                'relationship_type': 'NO_MATCH',
                'scores': {  # Initialize empty scores even in error case
                    'keyword_score': 0.0,
                    'semantic_score': 0.0,
                    'llm_score': 0.0,
                    'final_score': 0.0
                },
                'error': str(e)
            }

    def get_unclassified_use_cases(self, limit: Optional[int] = None) -> List[dict]:
        """Fetch use cases without technology classifications"""
        query = """
        MATCH (u:UseCase)
        OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
        WHERE NOT EXISTS((u)-[:USES_TECHNOLOGY]->(:AICategory))
        RETURN {
            id: u.id,
            name: u.name,
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            agency: {
                name: CASE WHEN a IS NOT NULL THEN a.name ELSE 'Unknown Agency' END,
                abbreviation: CASE WHEN a IS NOT NULL THEN COALESCE(a.abbreviation, 'Unknown') ELSE 'Unknown' END
            }
        } as use_case
        ORDER BY use_case.id
        """
        if limit:
            query += f" LIMIT {limit}"
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record["use_case"]) for record in result]

    def get_technology_categories(self) -> List[dict]:
        """Fetch all technology categories with their properties"""
        query = """
        MATCH (c:AICategory)
        WITH c
        // Get keywords
        OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.type IN ['technical_keywords', 'capabilities', 'business_language']
        WITH c, 
             collect(DISTINCT {
                 name: k.name,
                 type: k.type,
                 relevance_score: COALESCE(k.relevance_score, 0.5)
             }) as keywords
        // Get capabilities
        OPTIONAL MATCH (c)-[:HAS_CAPABILITY]->(cap:Capability)
        WITH c, keywords,
             collect(DISTINCT cap.name) as capabilities
        // Get zone info
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        RETURN {
            id: c.id,
            name: c.name,
            description: c.category_definition,
            maturity_level: c.maturity_level,
            keywords: keywords,
            capabilities: capabilities,
            zone: CASE WHEN z IS NOT NULL THEN z.name ELSE null END
        } as category
        ORDER BY c.name
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            categories = [dict(record["category"]) for record in result]
            if not categories:
                raise ValueError("No technology categories found in the database")
            return categories

    def _determine_relationship_type(self, final_score: float) -> str:
        """Determine relationship type based on score thresholds."""
        if final_score >= Config.Scoring.THRESHOLDS['primary']:
            return 'PRIMARY'
        elif final_score >= Config.Scoring.THRESHOLDS['secondary']:
            return 'SECONDARY'
        elif final_score >= Config.Scoring.THRESHOLDS['related']:
            return 'RELATED'
        return 'NO_MATCH'

class TextProcessor:
    """Handle text processing and embedding operations"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def cleanup_text(self, text: str, preserve_case: bool = False) -> str:
        """Standardize text for better matching"""
        if not text:
            return ""
            
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
        
    def get_keyword_matches(self, 
                          text: str, 
                          keywords: List[str], 
                          threshold: float = 0.3) -> Tuple[float, Set[str]]:
        """Enhanced keyword matching with partial matching and relevance scoring"""
        if not text or not keywords:
            return 0.0, set()
            
        # Clean and standardize text
        text = self.cleanup_text(text)
        matched_keywords = set()
        total_score = 0.0
        
        for keyword in keywords:
            keyword = self.cleanup_text(keyword)
            
            # Exact match
            if keyword in text:
                matched_keywords.add(keyword)
                total_score += 1.0
                continue
                
            # Handle compound terms
            keyword_parts = keyword.split()
            if len(keyword_parts) > 1:
                # Check if all parts appear in text in any order
                if all(part in text for part in keyword_parts):
                    matched_keywords.add(keyword)
                    total_score += 0.8
                    continue
                    
                # Check if most parts appear (for longer compound terms)
                matched_parts = sum(1 for part in keyword_parts if part in text)
                if matched_parts / len(keyword_parts) >= 0.7:
                    matched_keywords.add(keyword)
                    total_score += 0.6
                    continue
            
            # Partial matching for single terms
            if len(keyword) > 4:  # Only for longer terms
                # Check for substring match with minimum length
                min_length = max(4, len(keyword) * threshold)
                if any(keyword[i:i+len(keyword)] in text 
                      for i in range(len(keyword)) 
                      if len(keyword[i:i+len(keyword)]) >= min_length):
                    matched_keywords.add(keyword)
                    total_score += 0.4
        
        # Normalize score
        if keywords:
            final_score = min(1.0, total_score / len(keywords))
        else:
            final_score = 0.0
            
        return round(final_score, 3), matched_keywords
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text with improved preprocessing"""
        # Clean and standardize text
        text = self.cleanup_text(text)
        
        # Handle empty or invalid text
        if not text:
            return self.model.encode("", convert_to_tensor=True)
            
        return self.model.encode(text, convert_to_tensor=True)
        
    def calculate_semantic_similarity(self, 
                                   vec1: np.ndarray,
                                   vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors with validation"""
        try:
            similarity = float(util.pytorch_cos_sim(vec1, vec2)[0][0])
            return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1
        except Exception as e:
            logging.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
            
    def calculate_weighted_semantic_score(self,
                                       use_case_text: str,
                                       category_text: str,
                                       field_weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate semantic similarity with field weighting"""
        if not field_weights:
            field_weights = {
                'description': 0.4,
                'purpose_benefits': 0.3,
                'outputs': 0.2,
                'systems': 0.1
            }
            
        # Get embeddings for each field
        use_case_fields = use_case_text.split('\n')
        category_fields = category_text.split('\n')
        
        total_score = 0.0
        total_weight = 0.0
        
        for field, weight in field_weights.items():
            if field in use_case_fields and field in category_fields:
                vec1 = self.get_embedding(use_case_fields[field])
                vec2 = self.get_embedding(category_fields[field])
                score = self.calculate_semantic_similarity(vec1, vec2)
                total_score += score * weight
                total_weight += weight
                
        if total_weight > 0:
            return round(total_score / total_weight, 3)
        return 0.0

class LLMProcessor:
    """Handle LLM-based verification using OpenAI"""
    
    def __init__(self):
        self.setup_client()
    
    def setup_client(self):
        """Setup OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        openai.api_key = api_key
        self.model = Config.LLM.MODEL
        self.client = openai.Client()
    
    def verify_match(self,
                    use_case_text: str,
                    category_text: str) -> Tuple[float, str]:
        """Verify match using OpenAI"""
        prompt = f"""You are a strict JSON-only response system performing AI technology category matching.
        Evaluate if this use case matches the category based on the provided information.
        
        Format your response as a JSON object with ONLY these fields:
        {{
            "confidence": float between 0 and 1,
            "justification": "string explanation"
        }}
        
        Use Case: {use_case_text}
        Category: {category_text}
        
        Evaluate if this use case matches this category and return ONLY the required JSON object."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.LLM.TEMPERATURE,
                max_tokens=Config.LLM.MAX_TOKENS,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            return float(result["confidence"]), result["justification"]
        except Exception as e:
            logging.error(f"LLM verification failed: {str(e)}")
            return 0.0, f"Error during LLM verification: {str(e)}"
    
    def analyze_unmatched_case(self,
                             use_case_text: str,
                             zones: List[str],
                             capabilities: List[str],
                             patterns: List[str]) -> Dict[str, Any]:
        """Analyze why a use case doesn't match any categories"""
        prompt = f"""You are a strict JSON-only response system analyzing unmatched AI use cases.
        Analyze why this use case doesn't match any existing categories and suggest improvements.
        
        Format your response as a JSON object with ONLY these fields:
        {{
            "reason_category": "NOVEL_TECH" | "IMPLEMENTATION_SPECIFIC" | "NON_AI" | "UNCLEAR_DESC" | "OTHER",
            "detailed_analysis": "string explanation of why no match was found",
            "improvement_suggestions": ["suggestion1", "suggestion2", ...],
            "potential_new_categories": ["category1", "category2", ...],
            "key_technologies": ["tech1", "tech2", ...],
            "closest_zone": "string name of most relevant zone",
            "missing_capabilities": ["capability1", "capability2", ...]
        }}
        
        Use Case: {use_case_text}
        
        Available Zones:
        {chr(10).join(f"- {zone}" for zone in zones)}
        
        Available Capabilities:
        {chr(10).join(f"- {cap}" for cap in capabilities)}
        
        Available Patterns:
        {chr(10).join(f"- {pattern}" for pattern in patterns)}
        
        Analyze the use case and return ONLY the required JSON object."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.LLM.TEMPERATURE,
                max_tokens=Config.LLM.MAX_TOKENS,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            # Ensure all required fields are present with defaults if missing
            return {
                "reason_category": result.get("reason_category", "UNCLEAR_DESC"),
                "detailed_analysis": result.get("detailed_analysis", "Analysis failed"),
                "improvement_suggestions": result.get("improvement_suggestions", []),
                "potential_new_categories": result.get("potential_new_categories", []),
                "key_technologies": result.get("key_technologies", []),
                "closest_zone": result.get("closest_zone", "Unknown"),
                "missing_capabilities": result.get("missing_capabilities", [])
            }
        except Exception as e:
            logging.error(f"Unmatched analysis failed: {str(e)}")
            return {
                "reason_category": "ERROR",
                "detailed_analysis": f"Error during analysis: {str(e)}",
                "improvement_suggestions": [],
                "potential_new_categories": [],
                "key_technologies": [],
                "closest_zone": "Unknown",
                "missing_capabilities": []
            }

async def verify_environment() -> bool:
    """Verify all required components of the environment are working."""
    checks_passed = True
    logging.info("\n=== Environment Verification ===")
    
    # 1. Check Environment Variables
    logging.info("\n1. Checking Environment Variables...")
    for var in Config.REQUIRED_ENV_VARS['base']:
        if not os.getenv(var):
            logging.error(f"❌ Missing required environment variable: {var}")
            checks_passed = False
        else:
            logging.info(f"✓ Found {var}")

    # 2. Check Neo4j Schema
    logging.info("\n2. Verifying Neo4j Schema...")
    try:
        schema = load_neo4j_schema()
        if not schema:
            logging.error("❌ Could not load Neo4j schema")
            checks_passed = False
        else:
            logging.info("✓ Successfully loaded Neo4j schema")
            for key in ['nodes', 'relationships', 'version']:
                if key not in schema:
                    logging.error(f"❌ Schema missing required key: {key}")
                    checks_passed = False
                else:
                    logging.info(f"✓ Schema contains {key}")
    except Exception as e:
        logging.error(f"❌ Error loading schema: {str(e)}")
        checks_passed = False

    # 3. Check Neo4j Connectivity
    logging.info("\n3. Testing Neo4j Connectivity...")
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            version = result.single()
            logging.info(f"✓ Connected to Neo4j: {version['name']} {version['versions'][0]}")
            
            constraints = session.run("SHOW CONSTRAINTS").data()
            if not constraints:
                logging.warning("⚠️ No constraints found in database")
            else:
                logging.info(f"✓ Found {len(constraints)} constraints")
        driver.close()
    except Exception as e:
        logging.error(f"❌ Neo4j connection failed: {str(e)}")
        checks_passed = False

    # 4. Check OpenAI
    logging.info("\n4. Testing OpenAI API...")
    try:
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        logging.info("✓ Successfully connected to OpenAI API")
        
        if Config.LLM.MODEL not in [m.id for m in models]:
            logging.warning(f"⚠️ Configured model {Config.LLM.MODEL} not found in available models")
        
        # Test format verification
        logging.info("\nTesting LLM response format...")
        test_prompt = """
        Analyze this test use case against a test category and provide your response in JSON format:
        
        Use Case: "An AI system that uses natural language processing to automatically categorize customer support tickets."
        Category: "Natural Language Processing - Systems that process and analyze human language data."
        
        Provide your response in JSON format with the following fields:
        {
            "confidence": float,  # 0-1 score of match confidence
            "relationship_type": "PRIMARY" | "SECONDARY" | "RELATED" | "NO_MATCH",
            "justification": string,  # Detailed explanation
            "key_alignments": [string],  # List of main points of alignment
            "potential_gaps": [string]  # List of any misalignments or gaps
        }
        """
        
        response = client.chat.completions.create(
            model=Config.LLM.MODEL,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=Config.LLM.TEMPERATURE,
            max_tokens=Config.LLM.MAX_TOKENS,
            response_format={ "type": "json_object" }
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            required_fields = ["confidence", "relationship_type", "justification", 
                             "key_alignments", "potential_gaps"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                logging.error(f"❌ LLM response missing required fields: {missing_fields}")
                checks_passed = False
            else:
                # Validate field types
                if not isinstance(result["confidence"], (int, float)):
                    logging.error("❌ LLM response 'confidence' field is not a number")
                    checks_passed = False
                if result["relationship_type"] not in ["PRIMARY", "SECONDARY", "RELATED", "NO_MATCH"]:
                    logging.error("❌ LLM response 'relationship_type' field has invalid value")
                    checks_passed = False
                if not isinstance(result["key_alignments"], list):
                    logging.error("❌ LLM response 'key_alignments' field is not a list")
                    checks_passed = False
                if not isinstance(result["potential_gaps"], list):
                    logging.error("❌ LLM response 'potential_gaps' field is not a list")
                    checks_passed = False
                
                if checks_passed:
                    logging.info("✓ LLM response format verified successfully")
                    logging.info(f"Sample confidence score: {result['confidence']}")
                    logging.info(f"Sample relationship type: {result['relationship_type']}")
            
        except json.JSONDecodeError:
            logging.error("❌ LLM response is not valid JSON")
            checks_passed = False
            
    except Exception as e:
        logging.error(f"❌ OpenAI API test failed: {str(e)}")
        checks_passed = False

    # 5. Check Sentence Transformer
    logging.info("\n5. Testing Sentence Transformer...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode("Test sentence", convert_to_tensor=True)
        logging.info("✓ Successfully loaded sentence transformer model")
    except Exception as e:
        logging.error(f"❌ Sentence transformer test failed: {str(e)}")
        checks_passed = False

    # Summary
    logging.info("\n=== Verification Summary ===")
    if checks_passed:
        logging.info("✓ All environment checks passed successfully!")
    else:
        logging.error("❌ Some environment checks failed. Please review the logs above.")
    
    return checks_passed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Federal Use Case AI Technology Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-n", "--number", type=int, help="Number of use cases to process")
    group.add_argument("-a", "--all", action="store_true", help="Process all unclassified use cases")
    group.add_argument("--verify-env", action="store_true", help="Verify environment setup")
    
    parser.add_argument("--dry-run", action="store_true", help="Run without making database changes")
    parser.add_argument("--batch-size", type=int, default=Config.Processing.DEFAULT_BATCH_SIZE,
                       help=f"Number of cases per batch (default: {Config.Processing.DEFAULT_BATCH_SIZE})")
    parser.add_argument("--compare-deprecated", action="store_true",
                       help="Compare results with deprecated classifier")
    
    return parser.parse_args()

async def compare_with_deprecated(use_case: dict, results: List[MatchResult]) -> None:
    """Compare classification results with deprecated classifier"""
    try:
        # Import deprecated classifier (assuming it's in the same directory)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "deprecated_classifier",
            "scripts/classification/deprecated/fed_use_case_ai_classifier.py"
        )
        deprecated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deprecated)
        
        # Run deprecated classifier
        deprecated_classifier = deprecated.FedUseCaseClassifier()
        deprecated_results = await deprecated_classifier.classify_use_case(use_case)
        
        # Compare results
        logging.info(f"\nComparison for Use Case {use_case['id']}:")
        logging.info("New Classifier Results:")
        for r in results:
            logging.info(f"- {r.category_id}: {r.relationship_type} (score: {r.final_score:.3f})")
        
        logging.info("\nDeprecated Classifier Results:")
        for r in deprecated_results:
            logging.info(f"- {r.category_id}: {r.relationship_type} (score: {r.score:.3f})")
        
        # Analyze differences
        new_matches = {r.category_id: r.relationship_type for r in results}
        old_matches = {r.category_id: r.relationship_type for r in deprecated_results}
        
        differences = []
        for cat_id in set(new_matches.keys()) | set(old_matches.keys()):
            if cat_id not in new_matches:
                differences.append(f"Category {cat_id}: Only in deprecated (as {old_matches[cat_id]})")
            elif cat_id not in old_matches:
                differences.append(f"Category {cat_id}: Only in new (as {new_matches[cat_id]})")
            elif new_matches[cat_id] != old_matches[cat_id]:
                differences.append(
                    f"Category {cat_id}: Relationship changed from {old_matches[cat_id]} to {new_matches[cat_id]}"
                )
        
        if differences:
            logging.info("\nDifferences Found:")
            for diff in differences:
                logging.info(f"- {diff}")
        else:
            logging.info("\nNo differences found between classifiers")
            
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")

def main():
    """Main entry point for the classifier"""
    parser = argparse.ArgumentParser(description='Federal Use Case Classifier')
    parser.add_argument('-n', '--num-cases', type=int, default=None,
                       help='Number of use cases to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without making changes to Neo4j')
    parser.add_argument('--verify-env', action='store_true',
                       help='Only verify environment and exit')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        # Initialize classifier
        classifier = Neo4jClassifier()
        
        # Load schema metadata
        classifier.load_schema_metadata()
        
        # Get unclassified use cases
        use_cases = classifier.get_unclassified_use_cases(limit=args.num_cases)
        logging.info(f"Found {len(use_cases)} unclassified use cases")
        
        # Get technology categories
        try:
            categories = classifier.get_technology_categories()
            logging.info(f"Found {len(categories)} technology categories")
            
            # Log available categories
            logging.info("\nAvailable Technology Categories:")
            for cat in categories:
                logging.info(f"  - {cat['id']}: {cat.get('name', 'Unnamed')} ({len(cat.get('keywords', []))} keywords)")
                
        except ValueError as e:
            logging.error(str(e))
            return 1
            
        if args.dry_run:
            logging.info("\n=== DRY RUN CLASSIFICATION PREVIEW ===")
            
        # Track statistics
        total_matches = 0
        match_types = {'PRIMARY': 0, 'SECONDARY': 0, 'RELATED': 0, 'NO_MATCH': 0}
        method_scores = {'keyword': [], 'semantic': [], 'llm': []}
            
        # Process each use case
        for use_case in use_cases:
            try:
                logging.info(f"\n{'='*80}")
                logging.info(f"Processing Use Case: {use_case.get('name', use_case['id'])}")
                
                # Add agency details with better formatting
                agency_info = "Unknown Agency"
                if use_case.get('agency'):
                    agency_abbr = use_case['agency'].get('abbreviation', '')
                    agency_name = use_case['agency'].get('name', '')
                    if agency_abbr and agency_name:
                        agency_info = f"{agency_abbr} ({agency_name})"
                    elif agency_abbr:
                        agency_info = agency_abbr
                    elif agency_name:
                        agency_info = agency_name
                
                logging.info(f"Agency: {agency_info}")
                
                # Show purpose/benefits and outputs if available
                purpose_benefits = use_case.get('purpose_benefits', '')
                if purpose_benefits:
                    logging.info(f"\nPurpose & Benefits: {purpose_benefits[:200]}...")
                
                outputs = use_case.get('outputs', '')
                if outputs:
                    logging.info(f"\nOutputs: {outputs[:200]}...")
                
                matches = []
                no_match_reasons = []
                
                # Test against each category
                for category in categories:
                    result = classifier.classify_use_case(use_case)
                    
                    # Track method scores
                    method_scores['keyword'].append(result['scores']['keyword_score'])
                    method_scores['semantic'].append(result['scores']['semantic_score'])
                    method_scores['llm'].append(result['scores']['llm_score'])
                    
                    if result['scores']['final_score'] >= 0.4:  # Minimum threshold
                        matches.append(result)
                    else:
                        no_match_reasons.append({
                            'category': category.get('name', category['id']),
                            'scores': result['scores'],
                            'keywords': result.get('matched_keywords', [])
                        })
                
                if matches:
                    total_matches += 1
                    logging.info(f"\nFound {len(matches)} matches:")
                    for match in matches:
                        relationship_type = "PRIMARY" if match['scores']['final_score'] >= 0.8 else \
                                         "SECONDARY" if match['scores']['final_score'] >= 0.6 else \
                                         "RELATED"
                        match_types[relationship_type] += 1
                        
                        category_name = next((c.get('name', c['id']) for c in categories if c['id'] == match['category_id']), match['category_id'])
                        
                        logging.info(f"\n  Category: {category_name}")
                        logging.info(f"  Relationship: {relationship_type}")
                        logging.info(f"  Scores:")
                        logging.info(f"    - Final Score: {match['scores']['final_score']:.3f}")
                        
                        # Enhanced keyword type reporting
                        if 'keyword_details' in match:
                            logging.info("\n  Keyword Matches by Type:")
                            for ktype, details in match['keyword_details']['type_matches'].items():
                                if details:
                                    score = match['keyword_details']['type_scores'][ktype]
                                    logging.info(f"    - {ktype} (score: {score:.3f}):")
                                    logging.info(f"      * {', '.join(details)}")
                        
                        logging.info(f"\n  Method Scores:")
                        logging.info(f"    - Keyword Score: {match['scores']['keyword_score']:.3f}")
                        logging.info(f"    - Semantic Score: {match['scores']['semantic_score']:.3f}")
                        logging.info(f"    - LLM Score: {match['scores']['llm_score']:.3f}")
                        
                        if match.get('llm_suggested_relationship'):
                            logging.info(f"\n  LLM Suggestion: {match['llm_suggested_relationship']}")
                            
                        if args.dry_run:
                            logging.info(f"\n  [DRY RUN] Would create USES_TECHNOLOGY relationship:")
                            logging.info(f"    - From: UseCase({use_case['id']})")
                            logging.info(f"    - To: AICategory({category_name})")
                            logging.info(f"    - Properties:")
                            logging.info(f"      * confidence_score: {match['scores']['final_score']}")
                            logging.info(f"      * relationship_type: {relationship_type}")
                            logging.info(f"      * match_method: hybrid")
                            logging.info(f"      * validated: false")
                else:
                    match_types['NO_MATCH'] += 1
                    logging.info("\nNo matches found - Analyzing unmatched case")
                    
                    # Log near-miss categories (scores > 0.2 but < 0.4)
                    near_misses = sorted(
                        [r for r in no_match_reasons if r['scores']['final_score'] > 0.2],
                        key=lambda x: x['scores']['final_score'],
                        reverse=True
                    )[:3]  # Top 3 near misses
                    
                    if near_misses:
                        logging.info("\nNear Miss Categories:")
                        for miss in near_misses:
                            logging.info(f"  - {miss['category']}:")
                            logging.info(f"    * Final Score: {miss['scores']['final_score']:.3f}")
                            logging.info(f"    * Keyword Score: {miss['scores']['keyword_score']:.3f}")
                            logging.info(f"    * Semantic Score: {miss['scores']['semantic_score']:.3f}")
                            logging.info(f"    * LLM Score: {miss['scores']['llm_score']:.3f}")
                            if miss['keywords']:
                                logging.info(f"    * Partial Keyword Matches: {', '.join(miss['keywords'])}")
                    
                    # Prepare context for unmatched analysis
                    context = {
                        'use_case_text': classifier._prepare_use_case_text(use_case),
                        'zones': classifier.schema_metadata['zones'],
                        'capabilities': classifier.schema_metadata['capabilities'],
                        'patterns': classifier.schema_metadata['patterns']
                    }
                    
                    analysis = classifier.llm_processor.analyze_unmatched_case(**context)
                    
                    if args.dry_run:
                        logging.info("\n  [DRY RUN] Would create UnmatchedAnalysis node:")
                        logging.info(f"    - Reason: {analysis['reason_category']}")
                        logging.info(f"    - Analysis: {analysis['detailed_analysis'][:200]}...")
                        logging.info(f"    - Suggestions: {', '.join(analysis['improvement_suggestions'])}")
                        logging.info(f"    - Potential New Categories: {', '.join(analysis['potential_new_categories'])}")
                        logging.info(f"    - Key Technologies: {', '.join(analysis['key_technologies'])}")
                        logging.info(f"    - Closest Zone: {analysis['closest_zone']}")
                        logging.info(f"    - Missing Capabilities: {', '.join(analysis['missing_capabilities'])}")
                    
            except Exception as e:
                logging.error(f"Error processing use case {use_case['id']}: {str(e)}")
                continue
                
        if args.dry_run:
            # Calculate and log statistics
            processed_cases = len(use_cases)
            logging.info("\n=== CLASSIFICATION STATISTICS ===")
            logging.info(f"Total Use Cases Processed: {processed_cases}")
            logging.info(f"Total Matches Found: {total_matches}")
            logging.info("\nMatch Type Distribution:")
            for match_type, count in match_types.items():
                percentage = (count / processed_cases) * 100
                logging.info(f"  - {match_type}: {count} ({percentage:.1f}%)")
                
            logging.info("\nScoring Method Performance:")
            for method, scores in method_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    above_threshold = sum(1 for s in scores if s >= Config.Scoring.WEIGHTS[method]['threshold'])
                    threshold_rate = (above_threshold / len(scores)) * 100
                    logging.info(f"  - {method.capitalize()}:")
                    logging.info(f"    * Average Score: {avg_score:.3f}")
                    logging.info(f"    * Above Threshold: {threshold_rate:.1f}%")
            
            logging.info("\n=== DRY RUN COMPLETED ===")
            logging.info("No changes were made to the database")
            
        return 0
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 