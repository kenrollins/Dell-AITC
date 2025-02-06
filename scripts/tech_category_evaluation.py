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
            # Initialize result
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
            
            # Step 1: Try keyword matching first
            keyword_score, matched_terms = self.get_keyword_matches(
                use_case.combined_text,
                category.keywords + category.capabilities
            )
            result.keyword_score = keyword_score
            result.matched_keywords = list(matched_terms)
            
            # If we have any keyword matches, prioritize them
            if keyword_score >= Config.Thresholds.KEYWORD and matched_terms:
                result.method = MatchMethod.KEYWORD
                result.confidence = keyword_score
                result.final_score = keyword_score * Config.METHOD_WEIGHTS['keyword']['score_weight']
                # Add boost for multiple keyword matches
                if len(matched_terms) > 1:
                    result.final_score = min(1.0, result.final_score * 1.25)  # 25% boost for multiple matches
                result.relationship_type = result.determine_relationship_type(result.final_score)
                result.explanation = f"Keyword match ({keyword_score:.3f}) with terms: {', '.join(matched_terms)}"
                return result
            
            # Step 2: Try semantic matching
            semantic_score = self.get_semantic_similarity(
                use_case.combined_text,
                category.combined_text
            )
            result.semantic_score = semantic_score
            result.semantic_details = f"Semantic similarity score: {semantic_score:.3f}"
            
            # If semantic score is good, use it
            if semantic_score >= Config.Thresholds.SEMANTIC:
                result.method = MatchMethod.SEMANTIC
                result.confidence = semantic_score
                result.final_score = semantic_score * Config.METHOD_WEIGHTS['semantic']['score_weight']
                # Add boost if there are any keyword matches
                if matched_terms:
                    result.final_score = min(1.0, result.final_score * 1.2)  # 20% boost if keywords support
                result.relationship_type = result.determine_relationship_type(result.final_score)
                result.explanation = f"Semantic match ({semantic_score:.3f})"
                if matched_terms:
                    result.explanation += f" with supporting keywords: {', '.join(matched_terms)}"
                return result
            
            # Step 3: Only try LLM if we have some signal
            should_try_llm = (
                (keyword_score >= 0.1) or  # Any meaningful keyword score
                (semantic_score >= 0.15)    # Any meaningful semantic score
            )
            
            if should_try_llm:
                llm_result = self._llm_match(use_case, category)
                result.llm_score = llm_result[1]
                result.llm_justification = llm_result[3]
            
            # Calculate final score
            result.final_score = result.calculate_final_score()
            
            # Determine the best method
            weighted_scores = [
                (MatchMethod.KEYWORD, keyword_score * Config.METHOD_WEIGHTS['keyword']['score_weight']),
                (MatchMethod.SEMANTIC, semantic_score * Config.METHOD_WEIGHTS['semantic']['score_weight']),
                (MatchMethod.LLM, result.llm_score * Config.METHOD_WEIGHTS['llm']['score_weight'])
            ]
            
            best_method, best_score = max(weighted_scores, key=lambda x: x[1])
            
            if best_score >= Config.Thresholds.RELATED:
                result.method = best_method
                result.confidence = best_score
                result.relationship_type = result.determine_relationship_type(result.final_score)
                
                # Build explanation
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
                        explanations.append(f"LLM says: {result.llm_justification}")
                
                result.explanation = " | ".join(explanations)
            else:
                result.method = MatchMethod.NO_MATCH
                result.explanation = (
                    f"No significant match found. Scores - "
                    f"Keyword: {keyword_score:.3f}, "
                    f"Semantic: {semantic_score:.3f}"
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
            prompt = f"""
            Evaluate if this federal use case matches the AI technology category.
            First provide a confidence score between 0 and 100, then explain your reasoning.
            Format your response as:
            Score: [0-100]
            Explanation: [your detailed explanation]
            
            Use Case: {use_case.name}
            Description: {use_case.combined_text}
            
            Category: {category.name}
            Description: {category.combined_text}
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
        KEYWORD = 0.15     # Lowered to be more lenient
        SEMANTIC = 0.20    # Lowered significantly
        LLM = 0.50        # Keep as is
        
        # Relationship type thresholds
        PRIMARY = 0.65     # Lowered to get more primary matches
        SECONDARY = 0.50   # Lowered to get more secondary matches
        RELATED = 0.35    # Keep as is for related matches
    
    # Method weights for final score calculation
    METHOD_WEIGHTS = {
        'keyword': {
            'score_weight': 1.4,    # Increased to strongly prioritize keywords
            'confidence': 'HIGH',
            'boost_factor': 0.25    # Increased boost for keywords
        },
        'semantic': {
            'score_weight': 0.9,    # Reduced to prefer keywords
            'confidence': 'MEDIUM',
            'boost_factor': 0.15
        },
        'llm': {
            'score_weight': 0.8,    # Keep reduced
            'confidence': 'VARIABLE',
            'boost_factor': 0.10
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
    """Create a detailed result entry"""
    return {
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
        'keyword_score': f"{result.keyword_score:.3f}",
        'semantic_score': f"{result.semantic_score:.3f}",
        'llm_score': f"{result.llm_score:.3f}",
        'final_score': f"{result.final_score:.3f}",
        'confidence': f"{result.confidence:.3f}",
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

def save_results_to_csv(
    results: List[MatchResult], 
    use_cases: Dict[str, UseCase],
    categories: Dict[str, Category],
    output_dir: Path,
    text_processor: TextProcessor  # Add text_processor parameter
) -> Tuple[str, str]:
    """Save match results to CSV files"""
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
            # Get LLM analysis for unmatched use case
            analysis = text_processor._analyze_unmatched_usecase(use_case)
            
            # Use the official agency abbreviation
            agency_abr = clean_text(use_case.abbreviation, preserve_case=True).upper() if use_case.abbreviation else ""
            
            # Format purpose benefits and outputs
            purpose_benefits = '; '.join(use_case.purposes) if use_case.purposes else ""
            outputs = '; '.join(use_case.outputs) if use_case.outputs else ""
            
            # Create single no-match entry with analysis
            best_scores = max(case_results, key=lambda x: x.final_score)
            match_details = (
                f"Best scores - Keyword: {best_scores.keyword_score:.2f}, "
                f"Semantic: {best_scores.semantic_score:.2f}, "
                f"LLM: {best_scores.llm_score:.2f}"
            )
            
            neo4j_preview.append({
                'agency': clean_text(use_case.agency, preserve_case=True).upper(),
                'abr': agency_abr,
                'topic_area': clean_text(use_case.topic_area, preserve_case=True),
                'use_case_name': clean_text(use_case.name, preserve_case=True),
                'purpose_benefits': clean_text(purpose_benefits),
                'outputs': clean_text(outputs),
                'category_name': 'NO MATCH',
                'relationship_type': 'no_match',
                'confidence': '0.0%',
                'match_method': 'no_match',
                'match_details': match_details,
                'match_scoring': analysis
            })
    
    # Save files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    detailed_file = output_dir / f"tech_eval_detailed_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(detailed_file, index=False, encoding='utf-8')
    
    preview_file = output_dir / f"tech_eval_neo4j_preview_{timestamp}.csv"
    pd.DataFrame(neo4j_preview).to_csv(preview_file, index=False, encoding='utf-8')
    
    logging.info(f"[+] Detailed results saved to: {detailed_file}")
    logging.info(f"[+] Neo4j preview saved to: {preview_file}")
    
    return str(detailed_file), str(preview_file)

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