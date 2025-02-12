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
from typing import List, Dict, Any, Optional
from ..classifier import Classifier
from ..database.management.verify_database import DatabaseVerifier
from ...config import get_settings
from neo4j import AsyncGraphDatabase

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

class BatchProcessor:
    """Handles batch processing of use cases with monitoring."""
    
    def __init__(
        self,
        batch_size: int = 20,
        min_confidence: float = 0.45,
        max_retries: int = 3,
        dry_run: bool = False,
        api_timeout: float = 60.0  # Add API timeout parameter
    ):
        """Initialize batch processor with configuration."""
        self.settings = get_settings()
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.api_timeout = api_timeout
        
        # Initialize metrics
        self.metrics = {
            'total_processed': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'retried_cases': 0,
            'high_confidence_matches': 0,
            'low_confidence_matches': 0,
            'no_matches': 0,
            'processing_times': [],
            'errors': [],
            'start_time': None,
            'end_time': None,
            'case_results': []  # Store detailed results for each case
        }
        
        # Initialize components
        self.classifier = None
        self.driver = None
        self.verifier = None
        
    async def initialize(self):
        """Initialize connections and components."""
        # Initialize Neo4j connection
        self.driver = AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password)
        )
        
        # Initialize classifier with timeout
        self.classifier = Classifier(dry_run=self.dry_run)
        await self.classifier.initialize(api_timeout=self.api_timeout)
        
        # Initialize database verifier
        self.verifier = DatabaseVerifier(
            self.settings.neo4j_uri,
            self.settings.neo4j_user,
            self.settings.neo4j_password
        )
        
    async def cleanup(self):
        """Cleanup connections and resources."""
        if self.classifier:
            await self.classifier.close()
        if self.driver:
            await self.driver.close()
        if self.verifier:
            self.verifier.close()
            
    async def get_unprocessed_batch(self) -> List[Dict[str, Any]]:
        """Get next batch of unprocessed use cases."""
        async with self.driver.session() as session:
            query = """
            MATCH (u:UseCase)
            WHERE NOT (u)-[:CLASSIFIED_AS]->(:AIClassification)
            RETURN u
            ORDER BY u.created_at
            LIMIT $batch_size
            """
            result = await session.run(query, batch_size=self.batch_size)
            return [record["u"] async for record in result]
            
    async def process_batch(self) -> Dict[str, Any]:
        """Process a batch of use cases with monitoring."""
        try:
            # Start timing
            self.metrics['start_time'] = datetime.now()
            
            # Get batch of unprocessed cases
            batch = await self.get_unprocessed_batch()
            if not batch:
                logger.info("No unprocessed use cases found")
                return self.metrics
                
            logger.info(f"Processing batch of {len(batch)} use cases")
            
            # Process each use case
            for use_case in batch:
                start_time = datetime.now()
                
                try:
                    # Classify use case
                    result = await self.classifier.classify_use_case(
                        use_case,
                        save_to_db=not self.dry_run
                    )
                    
                    # Store full case result
                    case_result = {
                        'use_case': use_case,
                        'match_type': result.get('match_type', 'NONE'),
                        'confidence': result.get('confidence', 0.0),
                        'category_name': result.get('category_name'),
                        'keyword_score': result.get('keyword_score', 0.0),
                        'semantic_score': result.get('semantic_score', 0.0),
                        'llm_score': result.get('llm_score', 0.0),
                        'matched_keywords': result.get('matched_keywords', []),
                        'match_details': result.get('match_details', {}),
                        'field_match_scores': result.get('field_match_scores', {}),
                        'matched_terms': result.get('matched_terms', {}),
                        'llm_explanation': result.get('llm_explanation', ''),
                        'alternative_matches': result.get('alternative_matches', [])
                    }
                    self.metrics['case_results'].append(case_result)
                    
                    # Update metrics
                    self.metrics['total_processed'] += 1
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.metrics['processing_times'].append(processing_time)
                    
                    # Check for true no-match (no category found by any method)
                    if not result.get('category_name'):
                        self.metrics['no_matches'] += 1
                    else:
                        # If we have a category, determine confidence level
                        confidence = max(
                            result.get('confidence', 0.0),
                            result.get('llm_score', 0.0)  # Include LLM score in confidence check
                        )
                        
                        if confidence >= 0.7:
                            self.metrics['high_confidence_matches'] += 1
                            self.metrics['successful_classifications'] += 1
                        elif confidence >= 0.45:
                            self.metrics['low_confidence_matches'] += 1
                            self.metrics['successful_classifications'] += 1
                        else:
                            self.metrics['no_matches'] += 1  # Low confidence without LLM match
                        
                except Exception as e:
                    logger.error(f"Error processing use case {use_case.get('id')}: {str(e)}")
                    self.metrics['failed_classifications'] += 1
                    self.metrics['errors'].append({
                        'use_case_id': use_case.get('id'),
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
            # Complete timing
            self.metrics['end_time'] = datetime.now()
            
            # Calculate summary metrics
            if self.metrics['processing_times']:
                self.metrics['avg_processing_time'] = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
                self.metrics['max_processing_time'] = max(self.metrics['processing_times'])
                self.metrics['min_processing_time'] = min(self.metrics['processing_times'])
                
            # Log summary
            self._log_batch_summary()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise
            
    def _log_batch_summary(self):
        """Log batch processing summary."""
        logger.info("\n=== Batch Processing Summary ===")
        logger.info(f"Total Processed: {self.metrics['total_processed']}")
        logger.info(f"Successful Classifications: {self.metrics['successful_classifications']}")
        logger.info(f"Failed Classifications: {self.metrics['failed_classifications']}")
        logger.info(f"High Confidence Matches: {self.metrics['high_confidence_matches']}")
        logger.info(f"Low Confidence Matches: {self.metrics['low_confidence_matches']}")
        logger.info(f"No Matches: {self.metrics['no_matches']}")
        
        if self.metrics['processing_times']:
            logger.info(f"Average Processing Time: {self.metrics['avg_processing_time']:.2f}s")
            logger.info(f"Max Processing Time: {self.metrics['max_processing_time']:.2f}s")
            logger.info(f"Min Processing Time: {self.metrics['min_processing_time']:.2f}s")
            
        if self.metrics['errors']:
            logger.info(f"Total Errors: {len(self.metrics['errors'])}")
            
        total_time = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        logger.info(f"Total Processing Time: {total_time:.2f}s") 