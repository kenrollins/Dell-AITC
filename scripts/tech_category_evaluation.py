#!/usr/bin/env python3
"""
Technology Category Evaluation Script

This script evaluates and validates the effectiveness of different methods for mapping
use cases to AI technology categories using Neo4j, OpenAI, and semantic analysis.

Results are consolidated into a single output file with validation metrics.
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
    bureau: Optional[str]
    topic_area: str
    dev_stage: str
    purpose_benefits: List[str]
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
        """Calculate final score based on method weights and potential boosts"""
        weights = Config.METHOD_WEIGHTS
        
        # Calculate base weighted scores
        keyword_weighted = self.keyword_score * weights['keyword']['score_weight']
        semantic_weighted = self.semantic_score * weights['semantic']['score_weight']
        llm_weighted = self.llm_score * weights['llm']['score_weight']
        
        # Calculate boost based on multiple strong signals
        boost = 0.0
        
        # Boost for keyword + semantic agreement
        if (self.keyword_score >= Config.Thresholds.KEYWORD and 
            self.semantic_score >= Config.Thresholds.SEMANTIC):
            boost += weights['keyword']['boost_factor']
        
        # Additional boost if LLM confirms
        if (self.llm_score >= Config.Thresholds.LLM and 
            (self.keyword_score >= Config.Thresholds.KEYWORD or 
             self.semantic_score >= Config.Thresholds.SEMANTIC)):
            boost += weights['llm']['boost_factor']
        
        # Take the maximum score and apply boost
        base_score = max(keyword_weighted, semantic_weighted, llm_weighted)
        final_score = min(1.0, base_score + boost)
        
        return round(final_score, 3)  # Round to 3 decimal places

    def determine_relationship_type(self, score: float) -> RelationType:
        """Determine relationship type based on final score"""
        score = round(score, 3)  # Ensure consistent decimal places
        if score >= Config.Thresholds.PRIMARY:
            return RelationType.PRIMARY
        elif score >= Config.Thresholds.SECONDARY:
            return RelationType.SECONDARY
        elif score >= Config.Thresholds.RELATED:
            return RelationType.RELATED
        return RelationType.NO_MATCH

class Neo4jInterface:
    """Interface for Neo4j database operations"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('tech_mapper.neo4j')
        
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
        params = {}
        if limit:
            params['limit'] = limit
            
        query = """
        MATCH (u:UseCase)
        OPTIONAL MATCH (u)-[:HAS_PURPOSE]->(p:PurposeBenefit)
        OPTIONAL MATCH (u)-[:PRODUCES]->(o:Output)
        OPTIONAL MATCH (u)-[:USES_SYSTEM]->(s:System)
        OPTIONAL MATCH (u)<-[:HAS_USE_CASE]-(a:Agency)
        OPTIONAL MATCH (a)-[:HAS_BUREAU]->(b:Bureau)
        WITH DISTINCT u,
             a.name as agency_name,
             b.name as bureau_name,
             collect(DISTINCT p.description) as purposes,
             collect(DISTINCT o.description) as outputs,
             collect(DISTINCT s.name) as systems
        RETURN 
            coalesce(u.id, toString(id(u))) as id,
            u.name as name,
            agency_name as agency,
            bureau_name as bureau,
            u.topic_area as topic_area,
            u.dev_stage as dev_stage,
            purposes,
            outputs,
            systems,
            u.name + ' ' +
            coalesce(u.description, '') + ' ' +
            reduce(s = '', x IN purposes | s + ' ' + x) + ' ' +
            reduce(s = '', x IN outputs | s + ' ' + x) as combined_text
        """ + (" LIMIT $limit" if limit else "")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                use_cases = {}
                
                for record in result:
                    use_case = UseCase(
                        id=record["id"],
                        name=record["name"],
                        agency=record["agency"],
                        bureau=record["bureau"],
                        topic_area=record["topic_area"],
                        dev_stage=record["dev_stage"],
                        purpose_benefits=record["purposes"],
                        outputs=record["outputs"],
                        systems=record["systems"],
                        combined_text=record["combined_text"]
                    )
                    use_cases[use_case.id] = use_case
                
                self.logger.info(f"[+] Loaded {len(use_cases)} use cases")
                return use_cases
                
        except Exception as e:
            self.logger.error(f"[-] Failed to fetch use cases: {str(e)}")
            raise

    def save_match_result(self, match: MatchResult) -> bool:
        """Save a match result to Neo4j"""
        query = """
        MATCH (u:UseCase {id: $use_case_id})
        MATCH (c:AICategory {id: $category_id})
        MERGE (u)-[r:MATCHES {
            method: $method,
            confidence: $confidence,
            explanation: $explanation
        }]->(c)
        """
        
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    use_case_id=match.use_case_id,
                    category_id=match.category_id,
                    method=match.method.value,
                    confidence=match.confidence,
                    explanation=match.explanation
                )
                return True
        except Exception as e:
            self.logger.error(f"[-] Failed to save match result: {str(e)}")
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
        """Find keyword matches in text"""
        if not text or not keywords:
            return 0.0, set()
            
        text = self.cleanup_text(text)
        matches = set()
        matched_details = {}
        
        # Track partial matches for more nuanced scoring
        partial_matches = 0
        
        for keyword in keywords:
            keyword = keyword.lower()
            
            # Exact match
            if keyword in text:
                matches.add(keyword)
                matched_details[keyword] = 'exact'
                continue
            
            # Check for partial matches
            keyword_terms = set(keyword.split())
            
            # If keyword has multiple terms, check for partial match
            if len(keyword_terms) > 1:
                matched_terms = [term for term in keyword_terms if term in text]
                
                # Require at least half the terms to match
                if len(matched_terms) >= len(keyword_terms) / 2:
                    matches.add(keyword)
                    partial_matches += len(matched_terms) / len(keyword_terms)
                    matched_details[keyword] = f'partial: {matched_terms}'
        
        # Log matching details
        if matched_details:
            self.logger.info(f"[+] Keyword Matches: {matched_details}")
        
        # Calculate score
        total_matches = len(matches) + (partial_matches * 0.7)
        score = min(1.0, total_matches / (len(keywords) * 0.5))
        
        return score, matches
    
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
    
    def get_semantic_similarity(
        self, 
        text1: str, 
        text2: str,
        threshold: float = 0.5
    ) -> float:
        """Calculate semantic similarity between texts"""
        try:
            # Get embeddings
            embedding1 = self.get_embedding(text1)
            embedding2 = self.get_embedding(text2)
            
            # Calculate similarity
            similarity = float(
                util.pytorch_cos_sim(embedding1, embedding2)[0][0]
            )
            
            # Log strong matches
            if similarity >= threshold:
                self.logger.info(
                    f"[+] Strong semantic match ({similarity:.2f})"
                )
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"[-] Semantic similarity failed: {str(e)}")
            return 0.0
    
    def evaluate_match(self, use_case: UseCase, category: Category) -> MatchResult:
        """Evaluate if a use case matches a category using all available methods"""
        try:
            # Initialize result with default values
            result = MatchResult(
                use_case_id=use_case.id,
                category_id=category.id,
                method=MatchMethod.NO_MATCH,
                keyword_score=0.0,
                semantic_score=0.0,
                llm_score=0.0,
                final_score=0.0,
                confidence=0.0,
                relationship_type=RelationType.NO_MATCH,
                matched_keywords=[],
                semantic_details="",
                llm_justification="",
                explanation="",
                error=None
            )
            
            # Step 1: Always try keyword matching
            keyword_score, matched_terms = self.get_keyword_matches(
                use_case.combined_text,
                category.keywords + category.capabilities
            )
            result.keyword_score = keyword_score
            result.matched_keywords = list(matched_terms)
            
            # Step 2: Always try semantic matching
            semantic_score = self.get_semantic_similarity(
                use_case.combined_text,
                category.combined_text
            )
            result.semantic_score = semantic_score
            result.semantic_details = f"Semantic similarity score: {semantic_score:.3f}"
            
            # Step 3: Try LLM if either previous score shows promise
            if (keyword_score >= Config.Thresholds.RELATED or 
                semantic_score >= Config.Thresholds.RELATED):
                llm_result = self._llm_match(use_case, category)
                result.llm_score = llm_result[1]
                result.llm_justification = llm_result[3]
            
            # Calculate final score using all methods
            result.final_score = result.calculate_final_score()
            
            # Determine best method based on weighted scores
            weighted_scores = [
                (MatchMethod.KEYWORD, keyword_score * Config.METHOD_WEIGHTS['keyword']['score_weight']),
                (MatchMethod.SEMANTIC, semantic_score * Config.METHOD_WEIGHTS['semantic']['score_weight']),
                (MatchMethod.LLM, result.llm_score * Config.METHOD_WEIGHTS['llm']['score_weight'])
            ]
            best_method, best_score = max(weighted_scores, key=lambda x: x[1])
            
            # Set final result details
            if best_score >= Config.Thresholds.RELATED:
                result.method = best_method
                result.confidence = best_score
                result.relationship_type = result.determine_relationship_type(result.final_score)
                
                # Build detailed explanation
                explanations = []
                if result.keyword_score > 0:
                    explanations.append(f"Keyword match ({result.keyword_score:.3f})")
                    if result.matched_keywords:
                        explanations.append(f"Matched terms: {', '.join(result.matched_keywords)}")
                if result.semantic_score > 0:
                    explanations.append(f"Semantic match ({result.semantic_score:.3f})")
                if result.llm_score > 0:
                    explanations.append(f"LLM match ({result.llm_score:.3f})")
                    if result.llm_justification:
                        explanations.append(f"LLM says: {result.llm_justification}")
                
                result.explanation = " | ".join(explanations)
            else:
                result.method = MatchMethod.NO_MATCH
                result.explanation = (
                    f"No significant match found. Scores - "
                    f"Keyword: {keyword_score:.3f}, "
                    f"Semantic: {semantic_score:.3f}, "
                    f"LLM: {result.llm_score:.3f}"
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
            prompt = f"""
            Evaluate if this federal use case matches the AI technology category.
            Respond with 'yes' or 'no' and explain why.
            
            Use Case: {use_case.name}
            Description: {use_case.combined_text}
            
            Category: {category.name}
            Description: {category.combined_text}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI expert helping classify federal use cases into AI technology categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.lower()
            if "yes" in answer:
                return True, 0.8, RelationType.PRIMARY, answer
            return False, 0.0, RelationType.NO_MATCH, answer
            
        except Exception as e:
            logging.error(f"[-] LLM matching failed: {str(e)}")
            return False, 0.0, RelationType.NO_MATCH, str(e)

# Configure logging
def setup_logging() -> None:
    """Configure logging to write to both file and console with proper formatting"""
    log_dir = Path("data/output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"tech_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')  # Simplified console output
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
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
    """Configuration settings"""
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
        KEYWORD = 0.15     # Lower threshold to catch more potential matches
        SEMANTIC = 0.25    # Lower semantic threshold for broader matching
        LLM = 0.4         # Lower LLM threshold to allow more confirmations
        
        # Relationship type thresholds (kept same as they determine final strength)
        PRIMARY = 0.7     
        SECONDARY = 0.5
        RELATED = 0.3
    
    # Method weights for final score calculation
    METHOD_WEIGHTS = {
        'keyword': {
            'score_weight': 1.0,
            'confidence': 'HIGH',
            'boost_factor': 0.1  # Boost when combined with other methods
        },
        'semantic': {
            'score_weight': 0.9,
            'confidence': 'MEDIUM',
            'boost_factor': 0.1
        },
        'llm': {
            'score_weight': 0.8,
            'confidence': 'VARIABLE',
            'boost_factor': 0.1
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

def clean_text(text: str) -> str:
    """Clean text for CSV output by removing extra whitespace and newlines"""
    if not text:
        return ""
    # Replace newlines and multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def save_results_to_csv(
    results: List[MatchResult], 
    use_cases: Dict[str, UseCase],
    categories: Dict[str, Category],
    output_dir: Path
) -> Tuple[str, str]:
    """
    Save match results to two CSV files:
    1. Detailed analysis file with all evaluation metrics
    2. Neo4j preview file with essential relationship data
    
    Returns tuple of (detailed_file_path, preview_file_path)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Detailed Analysis File
    detailed_results = []
    # 2. Neo4j Preview File
    neo4j_preview = []
    
    for result in results:
        # Get associated use case and category
        use_case = use_cases.get(result.use_case_id)
        category = categories.get(result.category_id)
        
        # Format scores to 3 decimal places for detailed file
        keyword_score = f"{result.keyword_score:.3f}"
        semantic_score = f"{result.semantic_score:.3f}"
        llm_score = f"{result.llm_score:.3f}"
        final_score = f"{result.final_score:.3f}"
        confidence = f"{result.confidence:.3f}"
        
        # 1. Detailed Analysis Data
        detailed_results.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'use_case_id': result.use_case_id,
            'use_case_name': clean_text(use_case.name if use_case else None),
            'agency': clean_text(use_case.agency if use_case else None),
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
            'keyword_score': keyword_score,
            'semantic_score': semantic_score,
            'llm_score': llm_score,
            'final_score': final_score,
            'confidence': confidence,
            'matched_keywords': ', '.join(result.matched_keywords) if result.matched_keywords else '',
            'semantic_details': clean_text(result.semantic_details) if result.semantic_details else '',
            'llm_justification': clean_text(result.llm_justification) if result.llm_justification else '',
            'explanation': clean_text(result.explanation),
            'error': clean_text(result.error) if result.error else ''
        })
        
        # 2. Neo4j Preview Data - Only include significant matches
        if result.method not in [MatchMethod.NO_MATCH, MatchMethod.ERROR]:
            # Format confidence as percentage with 1 decimal point
            confidence_pct = "100.0%" if result.confidence >= 1.0 else f"{result.confidence * 100:.1f}%"
            
            # Generate method-specific match details
            if result.method == MatchMethod.KEYWORD:
                # Format keyword matches by type (exact vs partial)
                exact_matches = []
                partial_matches = []
                for keyword in result.matched_keywords:
                    if keyword.lower() in use_case.combined_text.lower():
                        exact_matches.append(keyword)
                    else:
                        partial_matches.append(keyword)
                
                match_details = []
                if exact_matches:
                    match_details.append(f"Exact matches: {', '.join(exact_matches)}")
                if partial_matches:
                    match_details.append(f"Partial matches: {', '.join(partial_matches)}")
                
                match_details = (
                    f"Keywords ({result.keyword_score:.1f}): " + 
                    " | ".join(match_details)
                )

            elif result.method == MatchMethod.SEMANTIC:
                # Extract key phrases from use case and category texts
                use_case_summary = use_case.name
                category_summary = category.name
                
                match_details = (
                    f"Semantic ({result.semantic_score:.1f}): "
                    f"'{use_case_summary}' aligns with '{category_summary}' - "
                    f"Strong semantic similarity in purpose and capabilities"
                )

            elif result.method == MatchMethod.LLM:
                # Clean and format LLM justification
                justification = clean_text(result.llm_justification)
                if len(justification) > 200:  # Truncate long justifications
                    justification = justification[:197] + "..."
                
                match_details = (
                    f"LLM ({result.llm_score:.1f}): {justification}"
                )

            # Add combined scores if multiple methods contributed
            if result.method != MatchMethod.NO_MATCH:
                combined_details = []
                
                # Base method details
                if result.method == MatchMethod.KEYWORD:
                    base_details = f"Primary: Keyword matching ({result.keyword_score:.1f})"
                elif result.method == MatchMethod.SEMANTIC:
                    base_details = f"Primary: Semantic analysis ({result.semantic_score:.1f})"
                elif result.method == MatchMethod.LLM:
                    base_details = f"Primary: LLM evaluation ({result.llm_score:.1f})"
                combined_details.append(base_details)
                
                # Supporting methods
                support_details = []
                if result.method != MatchMethod.KEYWORD and result.keyword_score >= Config.Thresholds.KEYWORD:
                    support_details.append(f"keyword ({result.keyword_score:.1f})")
                if result.method != MatchMethod.SEMANTIC and result.semantic_score >= Config.Thresholds.SEMANTIC:
                    support_details.append(f"semantic ({result.semantic_score:.1f})")
                if result.method != MatchMethod.LLM and result.llm_score >= Config.Thresholds.LLM:
                    support_details.append(f"LLM ({result.llm_score:.1f})")
                
                if support_details:
                    combined_details.append("Supported by: " + ", ".join(support_details))
                
                # Add boost information if applicable
                if (result.keyword_score >= Config.Thresholds.KEYWORD and 
                    result.semantic_score >= Config.Thresholds.SEMANTIC):
                    boost_amount = Config.METHOD_WEIGHTS['keyword']['boost_factor'] + Config.METHOD_WEIGHTS['semantic']['boost_factor']
                    combined_details.append(
                        f"Method agreement boost: +{boost_amount:.1f} "
                        f"(keyword-semantic alignment)"
                    )
                
                if (result.llm_score >= Config.Thresholds.LLM and 
                    (result.keyword_score >= Config.Thresholds.KEYWORD or 
                     result.semantic_score >= Config.Thresholds.SEMANTIC)):
                    boost_amount = Config.METHOD_WEIGHTS['llm']['boost_factor']
                    combined_details.append(
                        f"LLM confirmation boost: +{boost_amount:.1f}"
                    )
                
                # Add final score if different from primary method score
                if abs(result.final_score - getattr(result, f"{result.method.value}_score")) > 0.01:
                    combined_details.append(
                        f"Final adjusted score: {result.final_score:.1f} "
                        f"(after method agreement boosts)"
                    )
                
                match_details += " || " + " | ".join(combined_details)

            neo4j_preview.append({
                'use_case_id': result.use_case_id,
                'use_case_name': clean_text(use_case.name if use_case else None),
                'category_name': clean_text(category.name if category else None),
                'relationship_type': result.relationship_type.value,
                'confidence': confidence_pct,
                'match_method': result.method.value,
                'match_details': match_details
            })
    
    # Save Detailed Analysis File
    detailed_file = output_dir / f"tech_eval_detailed_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(detailed_file, index=False, encoding='utf-8')
    
    # Save Neo4j Preview File
    preview_file = output_dir / f"tech_eval_neo4j_preview_{timestamp}.csv"
    pd.DataFrame(neo4j_preview).to_csv(preview_file, index=False, encoding='utf-8')
    
    logging.info(f"[+] Detailed results saved to: {detailed_file}")
    logging.info(f"[+] Neo4j preview saved to: {preview_file}")
    
    return str(detailed_file), str(preview_file)

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging first
    setup_logging()
    
    # Verify environment before proceeding
    env_status = verify_environment()
    if not all(env_status.values()):
        logging.error("Environment verification failed. Please check the logs and fix any issues.")
        return 1
    
    logging.info("[+] Environment verification successful. Proceeding with execution.")
    
    try:
        # Initialize components
        neo4j = Neo4jInterface(
            Config.NEO4J_URI,
            Config.NEO4J_USER,
            Config.NEO4J_PASSWORD
        )
        text_processor = TextProcessor()
        
        # Load categories and use cases
        categories = neo4j.fetch_categories()
        logging.info(f"[+] Loaded {len(categories)} AI technology categories")
        
        # Determine number of use cases to process
        limit = None if args.all else (args.number or Config.SAMPLE_SIZE)
        use_cases = neo4j.fetch_use_cases(limit=limit)
        logging.info(f"[+] Loaded {len(use_cases)} use cases for processing")
        
        # Process each use case
        total_matches = 0
        all_results = []  # Store all results for CSV output
        
        for use_case in tqdm(use_cases.values(), desc="Processing use cases"):
            case_matches = []
            
            # Evaluate against each category
            for category in categories.values():
                match_result = text_processor.evaluate_match(
                    use_case=use_case,
                    category=category
                )
                
                # Save all results to CSV
                all_results.append(match_result)
                
                # Only count and save significant matches to Neo4j
                if match_result.method not in [MatchMethod.NO_MATCH, MatchMethod.ERROR]:
                    if neo4j.save_match_result(match_result):
                        case_matches.append(match_result)
                        total_matches += 1
            
            # Log results for this use case
            if case_matches:
                logging.info(
                    f"[+] Found {len(case_matches)} matches for use case: {use_case.name}"
                )
        
        # Save results to both files
        output_dir = Path("data/output/results")
        detailed_file, preview_file = save_results_to_csv(
            all_results, 
            use_cases, 
            categories, 
            output_dir
        )
        
        logging.info(f"[+] Processing complete. Total matches found: {total_matches}")
        logging.info(f"[+] Detailed results saved to: {detailed_file}")
        logging.info(f"[+] Neo4j preview saved to: {preview_file}")
        return 0
        
    except Exception as e:
        logging.error(f"[-] Processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)