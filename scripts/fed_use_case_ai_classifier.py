#!/usr/bin/env python3
"""
Federal Use Case AI Technology Classifier

This script evaluates federal AI use cases from government agencies and classifies them 
against 14 AI technology categories using a multi-method approach combining keyword matching, 
semantic analysis, and LLM verification.

Features:
- Keyword-based matching using curated technology keywords
- Semantic similarity analysis using sentence transformers
- LLM-based verification using OpenAI's GPT models
- Confidence scoring with weighted multi-signal approach
- Relationship type classification (PRIMARY, SECONDARY, RELATED)
- Detailed evaluation metrics and justifications

Output Files:
1. fed_use_case_ai_classification_[timestamp].csv
   - Detailed evaluation results with scores and justifications
   - Used for Neo4j import and analysis

2. fed_use_case_ai_classification_preview_[timestamp].csv
   - Simplified view of classification results
   - Used for quick review and validation

Usage:
    python fed_use_case_ai_classifier.py [-n NUM_CASES | -a]
    
Arguments:
    -n, --number NUM_CASES    Number of use cases to process
    -a, --all                Process all use cases

Environment Variables:
    NEO4J_URI               Neo4j database URI
    NEO4J_USER             Neo4j username
    NEO4J_PASSWORD         Neo4j password
    OPENAI_API_KEY         OpenAI API key
"""

import argparse
import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum, auto
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import re
import psutil
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate and classify federal AI use cases against technology categories"
    )
    
    # Add mutually exclusive group for number of use cases
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n", "--number",
        type=int,
        help="Number of use cases to process"
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Process all use cases"
    )
    
    return parser.parse_args()

@dataclass
class Category:
    """Data structure for AI technology categories"""
    id: str
    name: str
    status: str
    maturity_level: str
    keywords: List[str]
    capabilities: List[str]
    combined_text: str
    zone: Optional[str] = None
    integration_patterns: Optional[List[str]] = None

@dataclass
class UseCase:
    """Data structure for use cases"""
    id: str
    name: str
    agency: str
    abbreviation: str  # Agency abbreviation
    bureau: Optional[str]
    topic_area: str
    dev_stage: str
    purposes: List[str]
    outputs: List[str]
    systems: List[str]
    combined_text: str

class MatchMethod(Enum):
    """Enumeration of different matching methods"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    LLM = "llm"
    NO_MATCH = "no_match"
    ERROR = "error"

class RelationType(Enum):
    """Enumeration of relationship types based on match strength"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    RELATED = "related"
    NO_MATCH = "no_match"

@dataclass
class MatchResult:
    """Data structure for match results"""
    use_case_id: str
    category_id: str
    method: MatchMethod
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    llm_score: float = 0.0
    final_score: float = 0.0
    confidence: float = 0.0
    relationship_type: RelationType = RelationType.NO_MATCH
    matched_keywords: Optional[List[str]] = None
    semantic_details: Optional[str] = None
    llm_justification: Optional[str] = None
    explanation: str = ""
    error: Optional[str] = None

    def calculate_final_score(self) -> float:
        """Calculate final score with enhanced method agreement boosting"""
        weights = Config.METHOD_WEIGHTS
        
        # Calculate base weighted scores
        keyword_weighted = self.keyword_score * weights['keyword']['score_weight']
        semantic_weighted = self.semantic_score * weights['semantic']['score_weight']
        llm_weighted = self.llm_score * weights['llm']['score_weight']
        
        # Enhanced boost calculation based on method agreement
        boost = 0.0
        
        # Strong boost for three-method agreement
        if (self.keyword_score >= Config.Thresholds.KEYWORD and 
            self.semantic_score >= Config.Thresholds.SEMANTIC and
            self.llm_score >= Config.Thresholds.LLM):
            boost += 0.30  # Increased boost for all methods agreeing
            
        # Medium boost for semantic + LLM agreement (most reliable combination)
        elif (self.semantic_score >= Config.Thresholds.SEMANTIC and 
              self.llm_score >= Config.Thresholds.LLM):
            boost += 0.20  # Increased for stronger two-method agreement
            
        # New: Medium-high boost for strong two-method agreement
        elif ((self.keyword_score >= 0.6 and self.semantic_score >= 0.6) or
              (self.keyword_score >= 0.6 and self.llm_score >= 0.6) or
              (self.semantic_score >= 0.6 and self.llm_score >= 0.6)):
            boost += 0.15  # New boost level for strong two-method agreement
            
        # Smaller boost for keyword + semantic agreement
        elif (self.keyword_score >= Config.Thresholds.KEYWORD and 
              self.semantic_score >= Config.Thresholds.SEMANTIC):
            boost += 0.10
            
        # Small boost for keyword + LLM agreement
        elif (self.keyword_score >= Config.Thresholds.KEYWORD and 
              self.llm_score >= Config.Thresholds.LLM):
            boost += 0.08
        
        # Additional confidence boost for strong individual signals
        if self.keyword_score >= 0.8:  # Very strong keyword match
            boost += 0.05
        if self.semantic_score >= 0.7:  # Strong semantic match
            boost += 0.05
        if self.llm_score >= 0.8:  # Strong LLM confidence
            boost += 0.05
        
        # Take the maximum base score and apply cumulative boost
        base_score = max(keyword_weighted, semantic_weighted, llm_weighted)
        final_score = min(1.0, base_score + boost)
        
        return round(final_score, 3)

    def determine_relationship_type(self, score: float) -> RelationType:
        """Enhanced relationship type determination with cross-validation"""
        score = round(score, 3)
        
        # Cross-validate results for more accurate relationship type assignment
        method_agreement = sum(1 for s in [self.keyword_score, self.semantic_score, self.llm_score] 
                             if s >= Config.Thresholds.RELATED)
        
        # Adjust thresholds based on method agreement
        if method_agreement >= 2:  # At least two methods agree
            if score >= Config.Thresholds.PRIMARY * 0.95:  # Slightly lower threshold
                return RelationType.PRIMARY
            elif score >= Config.Thresholds.SECONDARY * 0.95:
                return RelationType.SECONDARY
            elif score >= Config.Thresholds.RELATED * 0.95:
                return RelationType.RELATED
        else:  # Single method or no agreement - use stricter thresholds
            if score >= Config.Thresholds.PRIMARY * 1.05:  # Slightly higher threshold
                return RelationType.PRIMARY
            elif score >= Config.Thresholds.SECONDARY * 1.05:
                return RelationType.SECONDARY
            elif score >= Config.Thresholds.RELATED * 1.05:
                return RelationType.RELATED
        
        return RelationType.NO_MATCH

@dataclass
class NoMatchAnalysis:
    """Data structure for capturing detailed information about unmatched use cases"""
    timestamp: str
    use_case_id: str
    use_case_name: str
    agency: str
    abbreviation: str
    bureau: Optional[str]
    topic_area: str
    dev_stage: str
    purposes: List[str]
    outputs: List[str]
    best_scores: Dict[str, float]  # Scores from each method
    reason_category: str  # One of: NOVEL_TECH, IMPLEMENTATION_SPECIFIC, NON_AI, UNCLEAR_DESC, OTHER
    llm_analysis: str    # Full LLM analysis text
    improvement_suggestions: str  # Extracted from LLM analysis
    potential_categories: List[str]  # Potential new categories to consider

class Neo4jInterface:
    """Interface for Neo4j database operations"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('tech_mapper.neo4j')
        self._setup_schema()
    
    def _setup_schema(self):
        """Set up required indexes for IMPLEMENTS relationships and LLM analyses"""
        try:
            with self.driver.session() as session:
                # Create indexes for relationship properties
                relationship_indexes = [
                    ("implements_confidence_idx", "confidence"),
                    ("implements_match_method_idx", "match_method"),
                    ("implements_final_score_idx", "final_score"),
                    ("implements_relationship_type_idx", "relationship_type")
                ]
                
                for idx_name, prop in relationship_indexes:
                    session.run(f"""
                    CREATE INDEX {idx_name} IF NOT EXISTS
                    FOR ()-[r:IMPLEMENTS]-()
                    ON (r.{prop})
                    """)
                    self.logger.info(f"Created/verified index {idx_name} on IMPLEMENTS.{prop}")
                
                # Create indexes for LLMAnalysis nodes
                llm_indexes = [
                    ("llm_analysis_id_idx", "id"),
                    ("llm_analysis_type_idx", "analysis_type"),
                    ("llm_analysis_timestamp_idx", "timestamp"),
                    ("llm_analysis_reason_idx", "reason_category"),
                    ("llm_analysis_model_idx", "model")
                ]
                
                for idx_name, prop in llm_indexes:
                    session.run(f"""
                    CREATE INDEX {idx_name} IF NOT EXISTS
                    FOR (a:LLMAnalysis)
                    ON (a.{prop})
                    """)
                    self.logger.info(f"Created/verified index {idx_name} on LLMAnalysis.{prop}")
                    
        except Exception as e:
            self.logger.error(f"Failed to set up schema: {str(e)}")
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch_categories(self) -> Dict[str, Category]:
        """Fetch all AI technology categories with their related data"""
        query = """
        MATCH (c:AICategory)
        OPTIONAL MATCH (c)-[:TAGGED_WITH]->(k:Keyword)
        OPTIONAL MATCH (c)-[:HAS_CAPABILITY]->(cap:Capability)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        OPTIONAL MATCH (c)-[:INTEGRATES_VIA]->(p:IntegrationPattern)
        WITH c,
             collect(DISTINCT k.name) as keywords,
             collect(DISTINCT cap.name) as capabilities,
             collect(DISTINCT z.name)[0] as zone,
             collect(DISTINCT p.name) as integration_patterns,
             c.name + ' ' +
             coalesce(c.description, '') + ' ' +
             reduce(s = '', x IN collect(DISTINCT k.name) | s + ' ' + x) + ' ' +
             reduce(s = '', x IN collect(DISTINCT cap.name) | s + ' ' + x) as combined_text
        RETURN 
            c.id as id,
            c.name as name,
            c.status as status,
            c.maturity_level as maturity_level,
            keywords,
            capabilities,
            zone,
            integration_patterns,
            combined_text
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                categories = {}
                
                for record in result:
                    category = Category(
                        id=record["id"],
                        name=record["name"],
                        status=record["status"],
                        maturity_level=record["maturity_level"],
                        keywords=record["keywords"],
                        capabilities=record["capabilities"],
                        zone=record["zone"],
                        integration_patterns=record["integration_patterns"],
                        combined_text=record["combined_text"]
                    )
                    categories[category.id] = category
                
                self.logger.info(f"[+] Loaded {len(categories)} categories")
                return categories
                
        except Exception as e:
            self.logger.error(f"[-] Failed to fetch categories: {str(e)}")
            raise

    def fetch_use_cases(self, limit: Optional[int] = None) -> Dict[str, UseCase]:
        """Fetch use cases with their related data"""
        # Build query parameters
        params = {'limit': limit} if limit else {}
        self.logger.info(f"[+] Attempting to fetch {limit if limit else 'all'} use cases")
        
        # Query that properly handles LIMIT
        query = """
        MATCH (u:UseCase)
        WITH count(u) as total_count
        MATCH (u:UseCase)
        WITH u, total_count
        ORDER BY u.name
        """ + (f"LIMIT $limit" if limit else "") + """
        OPTIONAL MATCH (u)-[:HAS_PURPOSE]->(p:PurposeBenefit)
        OPTIONAL MATCH (u)-[:PRODUCES]->(o:Output)
        OPTIONAL MATCH (u)-[:USES_SYSTEM]->(s:System)
        OPTIONAL MATCH (u)<-[:HAS_USE_CASE]-(a:Agency)
        OPTIONAL MATCH (a)-[:HAS_BUREAU]->(b:Bureau)
        WITH DISTINCT u, total_count,
             a.name as agency_name,
             a.abbreviation as agency_abbreviation,
             b.name as bureau_name,
             collect(DISTINCT p.description) as purposes,
             collect(DISTINCT o.description) as outputs,
             collect(DISTINCT s.name) as systems
        RETURN 
            total_count,
            coalesce(u.id, toString(id(u))) as id,
            u.name as name,
            agency_name as agency,
            agency_abbreviation as abbreviation,
            bureau_name as bureau,
            u.topic_area as topic_area,
            u.dev_stage as dev_stage,
            purposes,
            outputs,
            systems,
            u.name + ' ' +
            reduce(s = '', x IN purposes | s + ' ' + x) + ' ' +
            reduce(s = '', x IN outputs | s + ' ' + x) as combined_text
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                use_cases = {}
                total_count = None
                
                for record in result:
                    if total_count is None:
                        total_count = record["total_count"]
                        self.logger.info(f"[+] Total use cases in database: {total_count}")
                    
                    use_case = UseCase(
                        id=record["id"],
                        name=record["name"],
                        agency=record["agency"],
                        abbreviation=record["abbreviation"],
                        bureau=record["bureau"],
                        topic_area=record["topic_area"],
                        dev_stage=record["dev_stage"],
                        purposes=record["purposes"],
                        outputs=record["outputs"],
                        systems=record["systems"],
                        combined_text=record["combined_text"]
                    )
                    use_cases[use_case.id] = use_case
                
                actual_count = len(use_cases)
                if limit and actual_count < limit:
                    self.logger.warning(f"[-] Only retrieved {actual_count} use cases out of requested {limit}")
                else:
                    self.logger.info(f"[+] Successfully retrieved {actual_count} use cases")
                
                return use_cases
                
        except Exception as e:
            self.logger.error(f"[-] Failed to fetch use cases: {str(e)}")
            raise

    def _validate_match_result(self, match: MatchResult) -> Tuple[bool, List[str]]:
        """Validate match result before saving to Neo4j"""
        errors = []
        
        # 1. Node existence validation
        try:
            with self.driver.session() as session:
                # Check UseCase node exists
                use_case_exists = session.run(
                    "MATCH (u:UseCase {id: $id}) RETURN count(u) > 0 as exists",
                    id=match.use_case_id
                ).single()["exists"]
                
                if not use_case_exists:
                    errors.append(f"UseCase node with id {match.use_case_id} not found")
                
                # Check AICategory node exists
                category_exists = session.run(
                    "MATCH (c:AICategory {id: $id}) RETURN count(c) > 0 as exists",
                    id=match.category_id
                ).single()["exists"]
                
                if not category_exists:
                    errors.append(f"AICategory node with id {match.category_id} not found")
        except Exception as e:
            errors.append(f"Node existence validation failed: {str(e)}")
            
        # 2. Property type validation
        try:
            # Validate numeric properties
            for prop, value in [
                ("confidence", match.confidence),
                ("final_score", match.final_score)
            ]:
                if not isinstance(value, (int, float)):
                    errors.append(f"{prop} must be numeric, got {type(value)}")
                elif not 0 <= value <= 1:
                    errors.append(f"{prop} must be between 0 and 1, got {value}")
            
            # Validate string properties
            if not isinstance(match.method.value, str):
                errors.append(f"match_method must be string, got {type(match.method.value)}")
            
            if not isinstance(match.relationship_type.value, str):
                errors.append(f"relationship_type must be string, got {type(match.relationship_type.value)}")
            
            if not isinstance(match.explanation, str):
                errors.append(f"explanation must be string, got {type(match.explanation)}")
                
        except Exception as e:
            errors.append(f"Property type validation failed: {str(e)}")
            
        # 3. Enum value validation
        valid_methods = {method.value for method in MatchMethod}
        if match.method.value not in valid_methods:
            errors.append(f"Invalid match_method: {match.method.value}")
            
        valid_types = {rel_type.value for rel_type in RelationType}
        if match.relationship_type.value not in valid_types:
            errors.append(f"Invalid relationship_type: {match.relationship_type.value}")
            
        # 4. Score consistency validation
        if match.relationship_type != RelationType.NO_MATCH:
            if match.confidence <= 0:
                errors.append(f"Non-NO_MATCH relationship requires confidence > 0, got {match.confidence}")
            if match.final_score <= 0:
                errors.append(f"Non-NO_MATCH relationship requires final_score > 0, got {match.final_score}")
                
        # 5. Method-specific validation
        if match.method == MatchMethod.KEYWORD and not match.matched_keywords:
            errors.append("Keyword match method requires matched_keywords")
            
        if match.method == MatchMethod.LLM and not match.llm_justification:
            errors.append("LLM match method requires llm_justification")
            
        return len(errors) == 0, errors

    def save_match_result(self, match: MatchResult) -> bool:
        """Save a match result to Neo4j with validation and proper typing"""
        # First validate the match result
        is_valid, validation_errors = self._validate_match_result(match)
        if not is_valid:
            self.logger.error("Match result validation failed:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            return False
            
        query = """
        // Ensure we're connecting the right node types
        MATCH (u:UseCase {id: $use_case_id})
        MATCH (c:AICategory {id: $category_id})
        
        // Create or update the relationship with all required properties
        MERGE (u)-[r:IMPLEMENTS]->(c)
        SET r = {
            match_method: $method,
            confidence: $confidence,
            final_score: $final_score,
            relationship_type: $relationship_type,
            explanation: $explanation,
            last_updated: datetime()
        }
        
        // Create LLM Analysis node if LLM was used
        WITH u, c, r
        WHERE $llm_used = true
        MERGE (a:LLMAnalysis {
            id: $analysis_id,
            timestamp: datetime(),
            analysis_type: 'match',
            score: $llm_score,
            justification: $llm_justification,
            model: 'gpt-3.5-turbo'
        })
        MERGE (r)-[:HAS_ANALYSIS]->(a)
        """
        
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    use_case_id=match.use_case_id,
                    category_id=match.category_id,
                    method=match.method.value,
                    confidence=float(match.confidence),
                    final_score=float(match.final_score),
                    relationship_type=match.relationship_type.value,
                    explanation=str(match.explanation),
                    llm_used=match.method == MatchMethod.LLM,
                    analysis_id=f"llm_match_{match.use_case_id}_{match.category_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    llm_score=float(match.llm_score) if match.llm_score else 0.0,
                    llm_justification=match.llm_justification if match.llm_justification else ""
                )
                return True
        except Exception as e:
            self.logger.error(f"[-] Failed to save match result: {str(e)}")
            return False

    def save_no_match_analysis(self, use_case_id: str, analysis: NoMatchAnalysis) -> bool:
        """Save no-match LLM analysis to Neo4j"""
        query = """
        MATCH (u:UseCase {id: $use_case_id})
        
        // Create LLM Analysis node for no-match case
        CREATE (a:LLMAnalysis {
            id: $analysis_id,
            timestamp: datetime(),
            analysis_type: 'no_match',
            reason_category: $reason_category,
            analysis_text: $analysis_text,
            improvement_suggestions: $improvement_suggestions,
            model: 'gpt-3.5-turbo'
        })
        
        // Create relationship between UseCase and Analysis
        CREATE (u)-[:HAS_NO_MATCH_ANALYSIS]->(a)
        
        // Create relationships to potential categories if any
        WITH a
        UNWIND $potential_categories as category_name
        OPTIONAL MATCH (c:AICategory {name: category_name})
        WITH a, c
        WHERE c IS NOT NULL
        CREATE (a)-[:SUGGESTS_CATEGORY]->(c)
        """
        
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    use_case_id=use_case_id,
                    analysis_id=f"llm_no_match_{use_case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    reason_category=analysis.reason_category,
                    analysis_text=analysis.llm_analysis,
                    improvement_suggestions=analysis.improvement_suggestions,
                    potential_categories=analysis.potential_categories
                )
                return True
        except Exception as e:
            self.logger.error(f"[-] Failed to save no-match analysis: {str(e)}")
            return False

class TextProcessor:
    """Handles text processing and matching with memory management"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger('tech_mapper.text')
        self.embedding_cache = {}
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def cleanup_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase and normalize whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_keyword_matches(
        self, 
        text: str, 
        keywords: List[str], 
        threshold: float = 0.3
    ) -> Tuple[float, Set[str]]:
        """Find keyword matches in text with stricter partial matching"""
        try:
            if not text or not keywords:
                self.logger.debug("Empty text or keywords provided")
                return 0.0, set()
                
            text = self.cleanup_text(text)
            matches = set()
            matched_details = {}
            
            # Track partial matches for more nuanced scoring
            partial_matches = 0
            
            # Log input for debugging
            self.logger.debug(f"Processing text: {text[:100]}...")
            self.logger.debug(f"Keywords to match: {keywords}")
            
            for keyword in keywords:
                try:
                    if not keyword:
                        continue
                        
                    keyword = self.cleanup_text(keyword)
                    if not keyword:
                        continue
                    
                    # Exact match
                    if keyword in text:
                        matches.add(keyword)
                        matched_details[keyword] = 'exact'
                        self.logger.debug(f"Found exact match: {keyword}")
                        continue
                    
                    # Check for partial matches
                    keyword_terms = set(keyword.split())
                    if not keyword_terms:
                        continue
                    
                    # If keyword has multiple terms, check for partial match
                    if len(keyword_terms) > 1:
                        matched_terms = [term for term in keyword_terms if term and term in text]
                        
                        # Require at least 60% of terms to match
                        if matched_terms and len(matched_terms) >= len(keyword_terms) * 0.6:
                            matches.add(keyword)
                            partial_matches += len(matched_terms) / len(keyword_terms)
                            matched_details[keyword] = f'partial: {matched_terms}'
                            self.logger.debug(f"Found partial match for {keyword}: {matched_terms}")
                
                except Exception as e:
                    self.logger.warning(f"Error processing keyword '{keyword}': {str(e)}")
                    continue
            
            # Log matching details
            if matched_details:
                self.logger.info(f"[+] Keyword Matches: {matched_details}")
            
            # Calculate score with adjusted weights
            valid_keywords = [k for k in keywords if k and self.cleanup_text(k)]
            if not valid_keywords:
                return 0.0, set()
                
            total_matches = len(matches) + (partial_matches * 0.7)
            score = min(1.0, total_matches / (len(valid_keywords) * 0.4))
            
            self.logger.debug(f"Final score: {score:.3f} with {len(matches)} exact and {partial_matches:.1f} partial matches")
            return score, matches
            
        except Exception as e:
            self.logger.error(f"[-] Keyword matching failed: {str(e)}")
            return 0.0, set()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get or compute text embedding with caching"""
        if not text:
            return self.model.encode("")
        
        # Check cache
        text = self.cleanup_text(text)
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Compute new embedding
        try:
            embedding = self.model.encode(text)
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"[-] Failed to compute embedding: {str(e)}")
            return self.model.encode("")
    
    def calculate_semantic_score(self, use_case_text: str, tech_category_text: str) -> float:
        """Calculate semantic similarity score between use case and tech category"""
        try:
            # Get embeddings for both texts
            use_case_embedding = self.get_embedding(use_case_text)
            tech_category_embedding = self.get_embedding(tech_category_text)
            
            # Calculate cosine similarity - this already gives us values between -1 and 1
            similarity = cosine_similarity(
                np.array(use_case_embedding).reshape(1, -1),
                np.array(tech_category_embedding).reshape(1, -1)
            )[0][0]
            
            # Only normalize negative values to 0, keep positive values as is
            normalized_score = max(0.0, similarity)
            
            return float(normalized_score)
        except Exception as e:
            self.logger.error(f"Error calculating semantic score: {str(e)}")
            return 0.0
    
    def evaluate_match(self, use_case: UseCase, category: Category) -> MatchResult:
        """Evaluate if a use case matches a category using all available methods with enhanced validation"""
        try:
            # Initialize result with default values
            result = MatchResult(
                use_case_id=use_case.id,
                category_id=category.id,
                method=MatchMethod.NO_MATCH
            )
            
            # Validate inputs
            if not use_case.combined_text or not category.combined_text:
                result.error = "Missing required text content"
                return result
            
            # Enhanced chatbot detection
            chatbot_indicators = [
                'chatbot', 'virtual assistant', 'conversational ai', 'dialogue system',
                'conversational agent', 'virtual agent', 'conversational interface',
                'interactive assistant', 'chat interface', 'messaging interface'
            ]
            
            is_chatbot = any(indicator in use_case.combined_text.lower() for indicator in chatbot_indicators)
            
            def validate_score(score: float, name: str) -> float:
                """Helper to validate and clean scores"""
                try:
                    score = float(score)
                    if score < 0 or score > 1:
                        self.logger.warning(f"Invalid {name} score {score}, clamping to [0,1]")
                        score = max(0, min(1, score))
                    return round(score, 3)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid {name} score {score}, defaulting to 0.0")
                    return 0.0
            
            # Step 1: Try keyword matching with error recovery and chatbot boost
            try:
                keyword_score, matched_terms = self.get_keyword_matches(
                    use_case.combined_text,
                    category.keywords + category.capabilities
                )
                
                # Apply chatbot-specific boost for NLP and Intelligent End-User Computing
                if is_chatbot and category.name in ['Natural Language Processing (NLP)', 'Intelligent End-User Computing']:
                    keyword_score = min(1.0, keyword_score * 1.25)  # 25% boost
                    
                result.keyword_score = validate_score(keyword_score, 'keyword')
                result.matched_keywords = list(matched_terms)
            except Exception as e:
                self.logger.warning(f"Keyword matching failed, falling back to other methods: {str(e)}")
                result.keyword_score = 0.0
                result.matched_keywords = []
            
            # Step 2: Try semantic matching with error recovery and chatbot context
            try:
                # Add chatbot context to semantic matching
                enhanced_use_case_text = use_case.combined_text
                if is_chatbot:
                    enhanced_use_case_text += " This system implements conversational AI capabilities for natural language interaction."
                    
                semantic_score = self.calculate_semantic_score(
                    enhanced_use_case_text,
                    category.combined_text
                )
                result.semantic_score = validate_score(semantic_score, 'semantic')
                result.semantic_details = f"Semantic similarity score: {result.semantic_score:.3f}"
            except Exception as e:
                self.logger.warning(f"Semantic matching failed, falling back to other methods: {str(e)}")
                result.semantic_score = 0.0
                result.semantic_details = f"Semantic matching failed: {str(e)}"
            
            # Determine if LLM verification is needed
            should_try_llm = (
                (result.semantic_score >= 0.6 and not result.matched_keywords) or
                (result.semantic_score >= 0.3 and result.semantic_score < 0.6 and result.matched_keywords) or
                (result.keyword_score >= 0.3 and result.semantic_score < 0.3) or
                (result.keyword_score >= 0.8 or result.semantic_score >= 0.8)  # Always verify very strong matches
            )
            
            # Step 3: Try LLM with retries and error handling
            if should_try_llm:
                max_retries = 2
                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        llm_result = self._llm_match(use_case, category)
                        result.llm_score = validate_score(llm_result[1], 'llm')
                        result.llm_justification = llm_result[3]
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count <= max_retries:
                            self.logger.warning(f"LLM attempt {retry_count} failed, retrying: {str(e)}")
                            time.sleep(1)  # Brief delay before retry
                        else:
                            self.logger.error(f"LLM matching failed after {max_retries} attempts: {str(e)}")
                            result.llm_score = 0.0
                            result.llm_justification = f"LLM analysis failed: {str(e)}"
            
            # Calculate final score with validation
            result.final_score = validate_score(result.calculate_final_score(), 'final')
            
            # Determine best method and relationship type
            weighted_scores = [
                (MatchMethod.KEYWORD, result.keyword_score * Config.METHOD_WEIGHTS['keyword']['score_weight']),
                (MatchMethod.SEMANTIC, result.semantic_score * Config.METHOD_WEIGHTS['semantic']['score_weight']),
                (MatchMethod.LLM, result.llm_score * Config.METHOD_WEIGHTS['llm']['score_weight'])
            ]
            
            best_method, best_score = max(weighted_scores, key=lambda x: x[1])
            
            # Validate match criteria
            has_valid_match = (
                (result.keyword_score >= Config.Thresholds.KEYWORD) or
                (result.semantic_score >= Config.Thresholds.SEMANTIC) or
                (result.llm_score >= Config.Thresholds.LLM)
            )
            
            if has_valid_match and best_score >= Config.Thresholds.RELATED:
                result.method = best_method
                result.confidence = validate_score(best_score, 'confidence')
                result.relationship_type = result.determine_relationship_type(result.final_score)
                
                # Build comprehensive explanation
                explanations = []
                if result.keyword_score >= Config.Thresholds.KEYWORD:
                    explanations.append(f"Keyword match ({result.keyword_score:.3f})")
                    if result.matched_keywords:
                        explanations.append(f"Matched terms: {', '.join(result.matched_keywords)}")
                if result.semantic_score >= Config.Thresholds.SEMANTIC:
                    explanations.append(f"Semantic match ({result.semantic_score:.3f})")
                if result.llm_score >= Config.Thresholds.LLM:
                    explanations.append(f"LLM match ({result.llm_score:.3f})")
                    if result.llm_justification:
                        explanations.append(f"LLM analysis: {result.llm_justification}")
                
                result.explanation = " | ".join(explanations)
            else:
                result.method = MatchMethod.NO_MATCH
                result.relationship_type = RelationType.NO_MATCH
                result.confidence = 0.0
                result.explanation = (
                    f"No significant match found. Best scores - "
                    f"Keyword: {result.keyword_score:.3f}, "
                    f"Semantic: {result.semantic_score:.3f}"
                    + (f", LLM: {result.llm_score:.3f}" if result.llm_score > 0 else "")
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Match evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            return MatchResult(
                use_case_id=use_case.id,
                category_id=category.id,
                method=MatchMethod.ERROR,
                error=error_msg,
                explanation=error_msg
            )
            
    def _get_relationship_type(self, score: float) -> RelationType:
        """Determine relationship type based on score thresholds"""
        if score >= Config.Thresholds.PRIMARY:
            return RelationType.PRIMARY
        elif score >= Config.Thresholds.SECONDARY:
            return RelationType.SECONDARY
        elif score >= Config.Thresholds.RELATED:
            return RelationType.RELATED
        return RelationType.NO_MATCH
    
    def _check_keywords(self, use_case: UseCase, category: Category) -> bool:
        """Check if any category keywords appear in the use case"""
        text = f"{use_case.name} {use_case.combined_text}".lower()
        return any(keyword.lower() in text for keyword in category.keywords)
    
    def _semantic_match(self, use_case: UseCase, category: Category) -> float:
        """Calculate semantic similarity between use case and category"""
        use_case_text = f"{use_case.name} {use_case.combined_text}"
        category_text = f"{category.name} {category.combined_text}"
        
        use_case_embedding = self.get_embedding(use_case_text)
        category_embedding = self.get_embedding(category_text)
        
        return float(use_case_embedding @ category_embedding.T)
    
    def _llm_match(self, use_case: UseCase, category: Category) -> tuple[bool, float, RelationType, str]:
        """Use LLM to evaluate match between use case and category"""
        try:
            # Check for chatbot indicators
            chatbot_indicators = [
                'chatbot', 'virtual assistant', 'conversational ai', 'dialogue system',
                'conversational agent', 'virtual agent', 'conversational interface',
                'interactive assistant', 'chat interface', 'messaging interface'
            ]
            is_chatbot = any(indicator in use_case.combined_text.lower() for indicator in chatbot_indicators)
            
            # Enhanced prompt with chatbot context
            prompt = f"""
            Evaluate if this federal use case matches the AI technology category.
            First provide a confidence score between 0 and 100, then explain your reasoning.
            
            {
                "For chatbot/conversational AI systems, consider both the natural language processing capabilities and the end-user interaction aspects."
                if is_chatbot else ""
            }
            
            Format your response as:
            Score: [0-100]
            Explanation: [your detailed explanation]
            
            Use Case: {use_case.name}
            Description: {use_case.combined_text}
            
            Category: {category.name}
            Description: {category.combined_text}
            
            {
                "Note: For chatbot systems, consider both primary capabilities (NLP) and secondary aspects (user interaction)."
                if is_chatbot else ""
            }
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI expert helping classify federal use cases into AI technology categories. Always provide a numerical confidence score followed by your explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.lower()
            
            # Extract score and explanation
            try:
                score_line = answer.split('\n')[0]
                score = float(re.search(r'score:\s*(\d+)', score_line).group(1)) / 100
                explanation = ' '.join(answer.split('\n')[1:])
                
                # Apply chatbot-specific boost for relevant categories
                if is_chatbot and category.name in ['Natural Language Processing (NLP)', 'Intelligent End-User Computing']:
                    score = min(1.0, score * 1.15)  # 15% boost
                    
            except:
                # Fallback to binary scoring if parsing fails
                score = 0.8 if "yes" in answer else 0.0
                explanation = answer
            
            if score >= Config.Thresholds.LLM:
                return True, score, RelationType.PRIMARY, explanation
            return False, score, RelationType.NO_MATCH, explanation
            
        except Exception as e:
            logging.error(f"[-] LLM matching failed: {str(e)}")
            return False, 0.0, RelationType.NO_MATCH, str(e)

    def _analyze_unmatched_usecase(self, use_case: UseCase) -> str:
        """Get LLM analysis for why a use case didn't match any categories"""
        try:
            prompt = f"""
            This federal use case did not match any of our AI technology categories.
            Choose one of these reasons and explain why:
            - Novel/Emerging Technology: Uses AI approaches not in our current categories
            - Implementation Specific: Too focused on specific implementation details rather than AI capabilities
            - Non-AI Technology: Might not be an AI use case
            - Unclear Description: Insufficient information to determine AI capabilities
            - Other: (specify reason)

            Then provide improvement suggestions:
            - What additional information would help better categorize this use case?
            - Are there any potential AI categories we should consider adding?
            - How could the use case description be enhanced?

            Use Case: {use_case.name}
            Description: {use_case.combined_text}

            Format your response as a brief analysis starting with the chosen reason, followed by suggestions.
            Keep the total response under 200 words.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI expert analyzing federal use cases that didn't match our AI technology categories. Be direct and concise in your analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )
            
            # Clean and format the response
            analysis = clean_text(response.choices[0].message.content)
            # Remove any "Reason:" or "Analysis:" prefixes
            analysis = re.sub(r'^(reason|analysis):\s*', '', analysis, flags=re.IGNORECASE)
            return analysis
            
        except Exception as e:
            logging.error(f"[-] Unmatched analysis failed: {str(e)}")
            return "Analysis failed due to error"

# Configure logging
def setup_logging() -> None:
    """Configure logging to write to both file and console with proper formatting"""
    log_dir = Path("data/output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"tech_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')  # Simplified console output
    
    # Set up file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler with more concise format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress some verbose loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Configuration class
class Config:
    """Configuration settings with optimized thresholds"""
    # Neo4j settings
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Processing settings
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100"))  # Number of use cases to process
    MIN_MEMORY_MB = int(os.getenv("MIN_MEMORY_MB", "1024"))  # Min required memory in MB (1GB)
    
    # Scoring thresholds for cascading evaluation
    class Thresholds:
        KEYWORD = 0.35     # Slightly increased for more precision
        SEMANTIC = 0.40    # Increased to ensure stronger semantic matches
        LLM = 0.45        # Slightly lowered to better utilize LLM insights
        
        # Relationship type thresholds adjusted for better distribution
        PRIMARY = 0.75     # Increased to make more selective
        SECONDARY = 0.45   # Lowered to capture more mid-range matches
        RELATED = 0.35    # Maintained for broad coverage
    
    # Method weights optimized based on performance analysis
    METHOD_WEIGHTS = {
        'keyword': {
            'score_weight': 1.2,    # Slightly reduced but still prioritized
            'confidence': 'HIGH',
            'boost_factor': 0.20    # Increased for strong keyword matches
        },
        'semantic': {
            'score_weight': 1.0,    # Increased for better balance
            'confidence': 'MEDIUM',
            'boost_factor': 0.15
        },
        'llm': {
            'score_weight': 0.9,    # Increased slightly
            'confidence': 'VARIABLE',
            'boost_factor': 0.12
        }
    }

def verify_environment() -> Dict[str, bool]:
    """Verify all required components are available and configured"""
    status = {
        "neo4j": False,
        "sentence_transformer": False,
        "openai": False,
        "memory": False
    }
    
    # Check Neo4j connection
    try:
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        status["neo4j"] = True
        logging.info("[+] Neo4j connection verified")
    except Exception as e:
        logging.error(f"[-] Neo4j connection failed: {str(e)}")
    
    # Check sentence transformer
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        status["sentence_transformer"] = True
        logging.info("[+] Sentence transformers model loaded")
    except Exception as e:
        logging.error(f"[-] Sentence transformers failed: {str(e)}")
    
    # Check OpenAI API
    try:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found")
        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        status["openai"] = True
        logging.info("[+] OpenAI API connection verified")
    except Exception as e:
        logging.error(f"[-] OpenAI API connection failed: {str(e)}")
    
    # Check available memory
    try:
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
        if available_memory >= Config.MIN_MEMORY_MB:
            status["memory"] = True
            logging.info(f"[+] Sufficient memory available ({available_memory:.0f}MB)")
        else:
            logging.error(f"[-] Insufficient memory: {available_memory:.0f}MB available")
    except Exception as e:
        logging.error(f"[-] Memory check failed: {str(e)}")
    
    return status

def clean_text(text: str, preserve_case: bool = False) -> str:
    """Clean text for CSV output by removing extra whitespace and newlines"""
    try:
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # Replace newlines and multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Remove any non-printable characters
        cleaned = ''.join(char for char in cleaned if char.isprintable())
        cleaned = cleaned.strip()
        return cleaned if preserve_case else cleaned.lower()
    except Exception as e:
        logging.error(f"[-] Text cleaning failed: {str(e)}")
        return ""

def format_agency_name(name: str) -> str:
    """Format agency name with proper capitalization"""
    if not name:
        return ""
    # Common lowercase words in agency names
    lowercase_words = {'of', 'the', 'and', 'in', 'on', 'at', 'to', 'for', 'with'}
    words = name.split()
    # Capitalize first and last word always, and all others except lowercase_words
    return ' '.join(
        word.title() if i == 0 or i == len(words)-1 or word.lower() not in lowercase_words
        else word.lower()
        for i, word in enumerate(words)
    )

def create_detailed_result(result: MatchResult, use_case: UseCase, category: Category) -> dict:
    """Create a detailed result entry with validated scores and cleaned text"""
    # Validate and fix scores
    scores = {
        'keyword_score': result.keyword_score,
        'semantic_score': result.semantic_score,
        'llm_score': result.llm_score,
        'final_score': result.final_score,
        'confidence': result.confidence
    }
    
    for key, value in scores.items():
        # Convert to float and validate range
        try:
            value = float(value)
            if value < 0 or value > 1:
                logging.warning(f"Invalid {key}: {value}, clamping to [0,1]")
                value = max(0, min(1, value))
            scores[key] = round(value, 3)
        except (ValueError, TypeError):
            logging.warning(f"Invalid {key} value: {value}, defaulting to 0.0")
            scores[key] = 0.0
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'use_case_id': result.use_case_id,
        'use_case_name': clean_text(use_case.name if use_case else None),
        'agency': clean_text(use_case.agency if use_case else None),
        'abbreviation': clean_text(use_case.abbreviation if use_case else None, preserve_case=True),
        'bureau': clean_text(use_case.bureau if use_case else None),
        'topic_area': clean_text(use_case.topic_area if use_case else None),
        'dev_stage': clean_text(use_case.dev_stage if use_case else None),
        'category_id': result.category_id,
        'category_name': clean_text(category.name if category else None),
        'category_status': clean_text(category.status if category else None),
        'category_maturity': clean_text(category.maturity_level if category else None),
        'category_zone': clean_text(category.zone if category else None),
        'match_method': result.method.value,
        'relationship_type': result.relationship_type.value,
        'keyword_score': f"{scores['keyword_score']:.3f}",
        'semantic_score': f"{scores['semantic_score']:.3f}",
        'llm_score': f"{scores['llm_score']:.3f}",
        'final_score': f"{scores['final_score']:.3f}",
        'confidence': f"{scores['confidence']:.3f}",
        'matched_keywords': ', '.join(result.matched_keywords) if result.matched_keywords else '',
        'semantic_details': clean_text(result.semantic_details) if result.semantic_details else '',
        'llm_justification': clean_text(result.llm_justification) if result.llm_justification else '',
        'explanation': clean_text(result.explanation),
        'error': clean_text(result.error) if result.error else ''
    }

def create_match_preview(result: MatchResult, use_case: UseCase, category: Category) -> dict:
    """Create a match preview entry"""
    # Handle confidence percentage
    confidence_pct = "100.0%" if result.confidence >= 1.0 else f"{result.confidence * 100:.1f}%"
    
    # Use the official agency abbreviation in uppercase
    agency_abr = clean_text(use_case.abbreviation, preserve_case=True).upper() if use_case and use_case.abbreviation else ""
    
    # Format agency name with proper capitalization
    agency_name = format_agency_name(use_case.agency) if use_case and use_case.agency else ""
    
    # Safely format purpose benefits and outputs
    purpose_benefits = '; '.join(use_case.purposes) if use_case and use_case.purposes else ""
    outputs = '; '.join(use_case.outputs) if use_case and use_case.outputs else ""
    
    # Generate method-specific match details
    if result.method == MatchMethod.KEYWORD and result.matched_keywords:
        # Handle keyword matches
        exact_matches = []
        partial_matches = []
        if use_case and use_case.combined_text:
            use_case_text = use_case.combined_text.lower()
            for k in result.matched_keywords:
                if k.lower() in use_case_text:
                    exact_matches.append(k)
                else:
                    partial_matches.append(k)
        
        match_details = []
        if exact_matches:
            match_details.append(f"Exact: {', '.join(exact_matches)}")
        if partial_matches:
            match_details.append(f"Partial: {', '.join(partial_matches)}")
        match_details = f"Keywords: {' | '.join(match_details)}"
    
    elif result.method == MatchMethod.SEMANTIC:
        match_details = (
            f"Semantic: Strong alignment between '{use_case.name if use_case else 'unknown'}' "
            f"and '{category.name if category else 'unknown'}' concepts"
        )
    
    elif result.method == MatchMethod.LLM:
        match_details = f"LLM Analysis: {clean_text(result.llm_justification) if result.llm_justification else 'No analysis available'}"
    
    else:
        match_details = "Match details not available"
    
    # Generate scoring information
    scoring_details = []
    primary_score = getattr(result, f"{result.method.value}_score", 0.0)
    scoring_details.append(f"Primary: {result.method.value.title()} ({primary_score:.2f})")
    
    support_scores = []
    if result.method != MatchMethod.KEYWORD and result.keyword_score >= Config.Thresholds.KEYWORD:
        support_scores.append(f"keyword ({result.keyword_score:.2f})")
    if result.method != MatchMethod.SEMANTIC and result.semantic_score >= Config.Thresholds.SEMANTIC:
        support_scores.append(f"semantic ({result.semantic_score:.2f})")
    if result.method != MatchMethod.LLM and result.llm_score >= Config.Thresholds.LLM:
        support_scores.append(f"LLM ({result.llm_score:.2f})")
    
    match_scoring = " | ".join(scoring_details + [f"Support: {', '.join(support_scores)}"] if support_scores else scoring_details)
    
    # Create the preview dictionary with safe value handling
    return {
        'agency': agency_name,
        'abr': agency_abr,
        'topic_area': clean_text(use_case.topic_area if use_case else ""),
        'use_case_name': clean_text(use_case.name if use_case else ""),
        'purpose_benefits': clean_text(purpose_benefits),
        'outputs': clean_text(outputs),
        'category_name': clean_text(category.name if category else ""),
        'relationship_type': result.relationship_type.value,
        'confidence': confidence_pct,
        'match_method': result.method.value,
        'match_details': clean_text(match_details),
        'match_scoring': clean_text(match_scoring)
    }

def save_no_match_analysis(
    analysis: NoMatchAnalysis,
    output_dir: Path,
    format: str = 'json'  # or 'csv'
) -> None:
    """Save no-match analysis to appropriate files"""
    # Use the base output directory for no-match analysis
    base_output_dir = Path("data/output")
    analysis_dir = base_output_dir / 'no_match_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories by reason category
    reason_dir = analysis_dir / analysis.reason_category.lower()
    reason_dir.mkdir(exist_ok=True)
    
    # Create monthly directory for organization
    month_dir = reason_dir / analysis.timestamp[:6]  # YYYYMM
    month_dir.mkdir(exist_ok=True)
    
    if format == 'json':
        # Save individual analysis as JSON
        file_path = month_dir / f"{analysis.use_case_id}_{analysis.timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)
    
    # Also append to monthly CSV for easy analysis
    csv_path = month_dir / f"no_match_summary_{analysis.timestamp[:6]}.csv"
    df = pd.DataFrame([asdict(analysis)])
    
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(csv_path, index=False, encoding='utf-8')

def parse_llm_analysis(analysis_text: str) -> Tuple[str, str, List[str]]:
    """Parse LLM analysis into structured components"""
    try:
        # Extract the main reason category
        reason_patterns = {
            'NOVEL_TECH': r'Novel/Emerging Technology',
            'IMPLEMENTATION_SPECIFIC': r'Implementation Specific',
            'NON_AI': r'Non-AI Technology',
            'UNCLEAR_DESC': r'Unclear Description',
            'OTHER': r'Other'
        }
        
        reason_category = 'OTHER'
        for category, pattern in reason_patterns.items():
            if re.search(pattern, analysis_text, re.IGNORECASE):
                reason_category = category
                break
        
        # Extract improvement suggestions
        suggestions_match = re.search(r'improvement suggestions:(.*?)(?=\n\n|$)', 
                                    analysis_text, 
                                    re.IGNORECASE | re.DOTALL)
        improvement_suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
        
        # Extract potential new categories
        categories_match = re.search(r'categories we should consider adding:(.*?)(?=\n|$)',
                                   analysis_text,
                                   re.IGNORECASE | re.DOTALL)
        potential_categories = []
        if categories_match:
            categories_text = categories_match.group(1)
            potential_categories = [c.strip() for c in categories_text.split(',') if c.strip()]
        
        return reason_category, improvement_suggestions, potential_categories
        
    except Exception as e:
        logging.error(f"Failed to parse LLM analysis: {str(e)}")
        return 'OTHER', '', []

def save_results_to_csv(
    results: List[MatchResult], 
    use_cases: Dict[str, UseCase],
    categories: Dict[str, Category],
    output_dir: Path,
    text_processor: TextProcessor
) -> Tuple[str, str]:
    """Save match results to CSV files with proper data validation"""
    # Group results by use case
    use_case_results = {}
    for result in results:
        if result.use_case_id not in use_case_results:
            use_case_results[result.use_case_id] = []
        use_case_results[result.use_case_id].append(result)
    
    # Process results
    detailed_results = []
    neo4j_preview = []
    
    for use_case_id, case_results in use_case_results.items():
        use_case = use_cases.get(use_case_id)
        if not use_case:
            continue
            
        # Check if any results are significant matches
        significant_matches = [r for r in case_results 
                             if r.relationship_type != RelationType.NO_MATCH]
        
        # Process each result for detailed file
        for result in case_results:
            category = categories.get(result.category_id)
            detailed_results.append(create_detailed_result(result, use_case, category))
        
        # For Neo4j preview, either show matches or single no-match entry
        if significant_matches:
            for match in significant_matches:
                category = categories.get(match.category_id)
                neo4j_preview.append(
                    create_match_preview(match, use_case, category)
                )
        else:
            # Get best scores from all results for this use case
            best_scores = {
                'keyword': max(r.keyword_score for r in case_results),
                'semantic': max(r.semantic_score for r in case_results),
                'llm': max(r.llm_score for r in case_results)
            }
            
            # Get LLM analysis for unmatched use case
            try:
                analysis_text = text_processor._analyze_unmatched_usecase(use_case)
                
                # Parse the analysis
                reason_category, improvements, potential_cats = parse_llm_analysis(analysis_text)
                
                # Create and save no-match analysis
                no_match_analysis = NoMatchAnalysis(
                    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    use_case_id=use_case.id,
                    use_case_name=use_case.name,
                    agency=use_case.agency,
                    abbreviation=use_case.abbreviation,
                    bureau=use_case.bureau,
                    topic_area=use_case.topic_area,
                    dev_stage=use_case.dev_stage,
                    purposes=use_case.purposes,
                    outputs=use_case.outputs,
                    best_scores=best_scores,
                    reason_category=reason_category,
                    llm_analysis=analysis_text,
                    improvement_suggestions=improvements,
                    potential_categories=potential_cats
                )
                
                save_no_match_analysis(no_match_analysis, output_dir)
                
                # Create no-match preview entry
                neo4j_preview.append({
                    'agency': clean_text(use_case.agency, preserve_case=True).upper(),
                    'abr': clean_text(use_case.abbreviation, preserve_case=True).upper() if use_case.abbreviation else "",
                    'topic_area': clean_text(use_case.topic_area, preserve_case=True),
                    'use_case_name': clean_text(use_case.name, preserve_case=True),
                    'purpose_benefits': '; '.join(use_case.purposes) if use_case.purposes else "",
                    'outputs': '; '.join(use_case.outputs) if use_case.outputs else "",
                    'category_name': 'NO MATCH',
                    'relationship_type': 'no_match',
                    'confidence': '0.0%',
                    'match_method': 'no_match',
                    'match_details': f"Best scores - Keyword: {best_scores['keyword']:.3f}, Semantic: {best_scores['semantic']:.3f}, LLM: {best_scores['llm']:.3f}",
                    'match_scoring': analysis_text
                })
                
            except Exception as e:
                logging.error(f"Failed to analyze unmatched use case {use_case.name}: {str(e)}")
                analysis_text = "Analysis failed due to error"
                
                # Still create a preview entry even if analysis fails
                neo4j_preview.append({
                    'agency': clean_text(use_case.agency, preserve_case=True).upper(),
                    'abr': clean_text(use_case.abbreviation, preserve_case=True).upper() if use_case.abbreviation else "",
                    'topic_area': clean_text(use_case.topic_area, preserve_case=True),
                    'use_case_name': clean_text(use_case.name, preserve_case=True),
                    'purpose_benefits': '; '.join(use_case.purposes) if use_case.purposes else "",
                    'outputs': '; '.join(use_case.outputs) if use_case.outputs else "",
                    'category_name': 'NO MATCH',
                    'relationship_type': 'no_match',
                    'confidence': '0.0%',
                    'match_method': 'error',
                    'match_details': 'Analysis failed',
                    'match_scoring': str(e)
                })
    
    # Create DataFrames with proper data validation
    try:
        # Create detailed results DataFrame
        detailed_df = pd.DataFrame(detailed_results)
        
        # Create preview DataFrame
        preview_df = pd.DataFrame(neo4j_preview)
        
        # Save files with UTF-8 encoding and proper error handling
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        detailed_file = output_dir / f"ai_tech_classification_neo4j_{timestamp}.csv"
        try:
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to save detailed results: {str(e)}")
            raise
        
        # Save preview results
        preview_file = output_dir / f"ai_tech_classification_preview_{timestamp}.csv"
        try:
            preview_df.to_csv(preview_file, index=False, encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to save preview results: {str(e)}")
            raise
        
        logging.info(f"[+] Detailed results saved to: {detailed_file}")
        logging.info(f"[+] Neo4j preview saved to: {preview_file}")
        
        return str(detailed_file), str(preview_file)
        
    except Exception as e:
        logging.error(f"Failed to create or save DataFrames: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set up logging first
        setup_logging()
        logging.info("[+] Starting technology category evaluation")
        
        # Verify environment before proceeding
        logging.info("[+] Verifying environment...")
        env_status = verify_environment()
        if not all(env_status.values()):
            logging.error("Environment verification failed. Please check the logs and fix any issues.")
            return 1
        
        logging.info("[+] Environment verification successful. Proceeding with execution.")
        
        try:
            # Initialize components
            logging.info("[+] Initializing Neo4j connection...")
            neo4j = Neo4jInterface(
                Config.NEO4J_URI,
                Config.NEO4J_USER,
                Config.NEO4J_PASSWORD
            )
            
            logging.info("[+] Initializing text processor...")
            text_processor = TextProcessor()
            
            # Load categories and use cases
            logging.info("[+] Loading categories...")
            categories = neo4j.fetch_categories()
            logging.info(f"[+] Loaded {len(categories)} AI technology categories")
            
            # Determine number of use cases to process
            limit = None if args.all else (args.number or Config.SAMPLE_SIZE)
            logging.info(f"[+] Fetching {limit if limit else 'all'} use cases...")
            use_cases = neo4j.fetch_use_cases(limit=limit)
            logging.info(f"[+] Loaded {len(use_cases)} use cases for processing")
            
            if not categories or not use_cases:
                logging.error("[-] Failed to load required data")
                logging.error(f"Categories loaded: {len(categories)}")
                logging.error(f"Use cases loaded: {len(use_cases)}")
                return 1
            
            # Initialize statistics
            stats = {
                'total_use_cases': len(use_cases),
                'use_cases_with_matches': 0,
                'use_cases_no_matches': 0,
                'total_matches': 0,
                'matches_by_type': {
                    'primary': 0,
                    'secondary': 0,
                    'related': 0
                },
                'matches_by_method': {
                    'keyword': 0,
                    'semantic': 0,
                    'llm': 0
                }
            }
            
            # Process each use case
            use_cases_with_matches = set()
            all_results = []  # Store all results for CSV output
            
            logging.info("[+] Starting use case processing...")
            for use_case in tqdm(use_cases.values(), desc="Processing use cases"):
                try:
                    case_matches = []
                    
                    # Evaluate against each category
                    for category in categories.values():
                        try:
                            match_result = text_processor.evaluate_match(
                                use_case=use_case,
                                category=category
                            )
                            
                            # Save all results to CSV
                            all_results.append(match_result)
                            
                            # Update statistics
                            if match_result.method not in [MatchMethod.NO_MATCH, MatchMethod.ERROR]:
                                case_matches.append(match_result)
                                stats['total_matches'] += 1
                                stats['matches_by_method'][match_result.method.value] += 1
                                stats['matches_by_type'][match_result.relationship_type.value] += 1
                                use_cases_with_matches.add(use_case.id)
                                
                                # Save to Neo4j if significant match
                                neo4j.save_match_result(match_result)
                                
                        except Exception as e:
                            logging.error(f"[-] Error processing category {category.name} for use case {use_case.name}: {str(e)}")
                            continue
                    
                    # Log results for this use case
                    if case_matches:
                        logging.info(
                            f"[+] Found {len(case_matches)} matches for use case: {use_case.name}"
                        )
                        
                except Exception as e:
                    logging.error(f"[-] Error processing use case {use_case.name}: {str(e)}")
                    continue
            
            # Update final statistics
            stats['use_cases_with_matches'] = len(use_cases_with_matches)
            stats['use_cases_no_matches'] = stats['total_use_cases'] - stats['use_cases_with_matches']
            
            # Save results to both files
            logging.info("[+] Saving results to CSV...")
            output_dir = Path("data/output/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            detailed_file, preview_file = save_results_to_csv(
                all_results, 
                use_cases, 
                categories, 
                output_dir,
                text_processor
            )
            
            # Print summary statistics
            logging.info("\n=== Processing Summary ===")
            logging.info(f"Total Use Cases Processed: {stats['total_use_cases']}")
            
            # Handle division by zero for percentages
            use_cases_with_matches_pct = (
                (stats['use_cases_with_matches'] / stats['total_use_cases'] * 100)
                if stats['total_use_cases'] > 0 else 0.0
            )
            use_cases_no_matches_pct = (
                (stats['use_cases_no_matches'] / stats['total_use_cases'] * 100)
                if stats['total_use_cases'] > 0 else 0.0
            )
            
            logging.info(f"Use Cases with Matches: {stats['use_cases_with_matches']} ({use_cases_with_matches_pct:.1f}%)")
            logging.info(f"Use Cases with No Matches: {stats['use_cases_no_matches']} ({use_cases_no_matches_pct:.1f}%)")
            logging.info(f"\nTotal Matches Found: {stats['total_matches']}")
            
            if stats['total_matches'] > 0:
                logging.info("\nMatches by Type:")
                for type_name, count in stats['matches_by_type'].items():
                    match_type_pct = (count / stats['total_matches'] * 100)
                    logging.info(f"- {type_name.title()}: {count} ({match_type_pct:.1f}%)")
                
                logging.info("\nMatches by Method:")
                for method_name, count in stats['matches_by_method'].items():
                    match_method_pct = (count / stats['total_matches'] * 100)
                    logging.info(f"- {method_name.title()}: {count} ({match_method_pct:.1f}%)")
            else:
                logging.info("\nNo matches found to analyze.")
                
            logging.info(f"\n[+] Detailed results saved to: {detailed_file}")
            logging.info(f"[+] Neo4j preview saved to: {preview_file}")
            return 0
            
        except Exception as e:
            logging.error(f"[-] Processing failed in main execution block: {str(e)}")
            import traceback
            logging.error(f"[-] Traceback: {traceback.format_exc()}")
            return 1
            
    except Exception as e:
        print(f"[-] Critical error in main function: {str(e)}")  # Use print since logging might not be set up
        import traceback
        print(f"[-] Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)