"""
Dell-AITC Batch Processing Service (v2.2)
Handles batch processing of use cases with monitoring and error recovery.

Usage:
    from backend.app.services.batch import BatchProcessor
    processor = BatchProcessor(batch_size=20)
    await processor.process_batch()
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from ..classifier import Classifier
from ..database.management.verify_database import DatabaseVerifier
from ...config import get_settings
from neo4j import AsyncGraphDatabase
import json
import time
from tqdm import tqdm

from backend.app.config import get_settings
from backend.app.services.ai.keyword_classifier import KeywordClassifier
from backend.app.services.database.verifier import DatabaseVerifier
from backend.app.services.ai.llm_analyzer import LLMAnalyzer

# Configure logging
log_dir = Path("logs/batch_processing")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'batch_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingPhase:
    """Enum-like class for processing phases"""
    LOCAL = "local_processing"
    LLM_VERIFY = "llm_verification"
    LLM_FULL = "llm_full_analysis"

class BatchProcessor:
    """Batch processor for use case classification"""
    
    def __init__(
        self,
        batch_size: int = 20,
        min_confidence: float = 0.6,
        dry_run: bool = False,
        skip_verification: bool = False
    ):
        """Initialize batch processor
        
        Args:
            batch_size: Number of use cases to process in each batch
            min_confidence: Minimum confidence threshold for classification
            dry_run: Whether to run in dry run mode (no database changes)
            skip_verification: Whether to skip database verification
        """
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.dry_run = dry_run
        self.skip_verification = skip_verification
        
        # Components
        self.classifier = None
        self.verifier = None
        self.llm_analyzer = None
        
        # Metrics
        self.metrics = {
            "total_processed": 0,
            "high_confidence": 0,
            "medium_verified": 0,
            "low_confidence": 0,
            "unmatched": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        
    async def initialize(self):
        """Initialize processor components"""
        logger.info("Initializing batch processor")
        
        try:
            # Get settings
            settings = get_settings()
            
            # Initialize classifier
            self.classifier = KeywordClassifier()
            await self.classifier.initialize()
            
            # Initialize LLM analyzer
            self.llm_analyzer = LLMAnalyzer()
            await self.llm_analyzer.initialize()
            
            # Initialize verifier if needed
            if not self.skip_verification:
                self.verifier = DatabaseVerifier(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password
                )
                await self.verifier.initialize()
                
            logger.info("Batch processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize batch processor: {str(e)}")
            raise
            
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up batch processor resources")
        try:
            if self.classifier:
                try:
                    await self.classifier.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up classifier: {str(e)}")
                    
            if self.verifier:
                try:
                    await self.verifier.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up verifier: {str(e)}")
                    
            if self.llm_analyzer:
                try:
                    await self.llm_analyzer.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up LLM analyzer: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    async def process_batch(self, use_cases: List[Dict]) -> None:
        """Process a batch of use cases
        
        Args:
            use_cases: List of use cases to process
        """
        logger.info(f"Processing batch of {len(use_cases)} use cases")
        
        for use_case in tqdm(use_cases, desc="Processing use cases"):
            try:
                # Extract text
                text = f"{use_case.get('name', '')} {use_case.get('description', '')} {use_case.get('purpose_benefits', '')} {use_case.get('outputs', '')}"
                if not text.strip():
                    logger.warning(f"Empty text for use case {use_case.get('id')}")
                    self.metrics["errors"] += 1
                    continue
                    
                # Classify text
                matches = await self.classifier.classify_text(
                    text,
                    min_confidence=self.min_confidence
                )
                
                if not matches:
                    logger.debug(f"No matches found for use case {use_case.get('id')}")
                    # Try LLM analysis for no matches
                    no_match_analysis = await self.llm_analyzer.analyze_no_match(use_case)
                    self.metrics["unmatched"] += 1
                    continue
                    
                # Process matches
                best_match = matches[0]
                if best_match.confidence >= 0.8:
                    # High confidence match
                    self.metrics["high_confidence"] += 1
                    if not self.dry_run:
                        await self.save_match(use_case, best_match)
                elif best_match.confidence >= 0.6:
                    # Medium confidence match
                    if self.skip_verification or await self._verify_match(use_case, best_match):
                        self.metrics["medium_verified"] += 1
                        if not self.dry_run:
                            await self.save_match(use_case, best_match)
                    else:
                        self.metrics["low_confidence"] += 1
                else:
                    # Low confidence match
                    self.metrics["low_confidence"] += 1
                    
                self.metrics["total_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing use case {use_case.get('id')}: {str(e)}")
                self.metrics["errors"] += 1
                
    async def _verify_match(self, use_case: Dict, match: Dict) -> bool:
        """Verify a match using the database verifier
        
        Args:
            use_case: Use case to verify
            match: Match result to verify
            
        Returns:
            True if match is verified, False otherwise
        """
        if self.skip_verification or not self.verifier:
            return True
            
        try:
            # Verify category exists
            if not await self.verifier.verify_category_exists(match.category.name):
                logger.warning(f"Category {match.category.name} does not exist")
                return False
                
            # Verify use case exists
            if not await self.verifier.verify_use_case_exists(use_case["id"]):
                logger.warning(f"Use case {use_case['id']} does not exist")
                return False
                
            # Verify no existing relationship
            if await self.verifier.verify_relationship_exists(
                use_case["id"],
                match.category.name
            ):
                logger.warning(f"Relationship already exists for {use_case['id']}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying match: {str(e)}")
            return False
            
    async def run(self, max_cases: Optional[int] = None):
        """Run the batch processor
        
        Args:
            max_cases: Maximum number of cases to process
        """
        logger.info(f"Starting batch processing with size {self.batch_size}")
        if self.dry_run:
            logger.info("DRY RUN MODE - No changes will be saved to database")
            
        # Initialize components
        await self.initialize()
        
        try:
            # Record start time
            self.metrics["start_time"] = time.time()
            
            # Get unclassified use cases
            use_cases = await self.classifier.get_unclassified_use_cases(limit=max_cases)
            logger.info(f"Found {len(use_cases)} unclassified use cases")
            
            # Process in batches
            for i in range(0, len(use_cases), self.batch_size):
                if max_cases and i >= max_cases:
                    break
                    
                batch = use_cases[i:i + self.batch_size]
                await self.process_batch(batch)
                
            # Record end time
            self.metrics["end_time"] = time.time()
            
            # Log final metrics
            duration = self.metrics["end_time"] - self.metrics["start_time"]
            logger.info(f"Batch processing completed in {duration:.2f} seconds")
            logger.info(f"Total processed: {self.metrics['total_processed']}")
            logger.info(f"High confidence: {self.metrics['high_confidence']}")
            logger.info(f"Medium verified: {self.metrics['medium_verified']}")
            logger.info(f"Low confidence: {self.metrics['low_confidence']}")
            logger.info(f"Unmatched: {self.metrics['unmatched']}")
            logger.info(f"Errors: {self.metrics['errors']}")
            
        finally:
            await self.cleanup() 