"""
Test script for LLM-based classification of federal use cases.

Usage:
    python scripts/test_llm_classification.py [-n NUM_CASES] [--dry-run] [--use-case-id ID] [--model MODEL] [--batch-size SIZE] [--checkpoint-file FILE] [--all] [--retry-no-match]
"""

import sys
from pathlib import Path
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from backend.app.services.llm_analyzer import LLMAnalyzer
from backend.app.services.database.neo4j_service import Neo4jService
from backend.app.config import get_settings

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'llm_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.processed_cases = self._load_checkpoint()
        
    def _load_checkpoint(self) -> set:
        """Load processed case IDs from checkpoint file."""
        try:
            if Path(self.checkpoint_file).exists():
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return set()
            
    def save_checkpoint(self, case_id: str):
        """Save a processed case ID to checkpoint file."""
        try:
            self.processed_cases.add(case_id)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(list(self.processed_cases), f)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def is_processed(self, case_id: str) -> bool:
        """Check if a case has been processed."""
        return case_id in self.processed_cases

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, httpx.ConnectError, httpx.TimeoutException))
)
async def get_single_use_case(db_service: Neo4jService, use_case_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific use case by ID with retry logic."""
    query = """
    MATCH (u:UseCase {id: $id})
    OPTIONAL MATCH (u)-[:BELONGS_TO]->(c:AICategory)
    OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
    RETURN {
        id: u.id,
        name: coalesce(u.name, 'Unnamed Use Case'),
        description: u.description,
        purpose_benefits: u.purpose_benefits,
        outputs: u.outputs,
        status: coalesce(u.status, 'Draft'),
        dev_stage: 'Unknown',
        dev_method: 'Unknown',
        business_terms: [],
        current_category: c.name,
        agency: CASE 
            WHEN a IS NOT NULL 
            THEN {
                name: a.name,
                abbreviation: COALESCE(a.abbreviation, 'Unknown')
            }
            ELSE {
                name: 'Unknown Agency',
                abbreviation: 'Unknown'
            }
        END,
        created_at: toString(datetime()),
        last_updated: toString(datetime())
    } as use_case
    """
    try:
        results = await db_service.run_query(query, {"id": use_case_id})
        return results[0]['use_case'] if results else None
    except Exception as e:
        logger.error(f"Error getting use case {use_case_id}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, httpx.ConnectError, httpx.TimeoutException))
)
async def process_single_case(use_case: Dict[str, Any], analyzer: LLMAnalyzer, 
                            categories: List[Dict[str, Any]], db_service: Neo4jService, 
                            dry_run: bool, checkpoint_mgr: Optional[CheckpointManager] = None) -> None:
    """Process a single use case."""
    case_start_time = datetime.now()
    
    # Skip if already processed
    if checkpoint_mgr and checkpoint_mgr.is_processed(use_case['id']):
        logger.info(f"Skipping already processed case: {use_case['id']}")
        return
            
    agency_info = use_case.get('agency', {'name': 'Unknown Agency', 'abbreviation': 'Unknown'})
    logger.info(f"\nProcessing use case: {use_case['name']}")
    logger.info(f"ID: {use_case['id']}")
    logger.info(f"Agency: {agency_info['name']} ({agency_info['abbreviation']})")
    logger.info("-" * 40)
    
    try:
        # Get LLM analysis with retry
        for attempt in range(3):
            try:
                results = await analyzer.analyze_use_case(use_case, categories)
                break
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == 2:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                await asyncio.sleep(5 * (attempt + 1))
        
        case_duration = (datetime.now() - case_start_time).total_seconds()
        logger.info(f"\nCase processing time: {case_duration:.2f} seconds")
        
        # Log results
        if primary := results.get("primary_match"):
            logger.info(f"Primary Match: {primary['category']} (confidence: {primary['confidence']:.2f})")
            logger.info(f"Reasoning: {primary['reasoning']}")
        
        if supporting := results.get("supporting_matches"):
            logger.info("\nSupporting Matches:")
            for match in supporting:
                logger.info(f"- {match['category']} (confidence: {match['confidence']:.2f})")
                logger.info(f"  Reasoning: {match['reasoning']}")
        
        if related := results.get("related_matches"):
            logger.info("\nRelated Matches:")
            for match in related:
                logger.info(f"- {match['category']} (confidence: {match['confidence']:.2f})")
                logger.info(f"  Reasoning: {match['reasoning']}")
        
        if no_match := results.get("no_match_analysis"):
            logger.info("\nNo Match Analysis:")
            logger.info(f"Reason: {no_match['reason']}")
            logger.info(f"Technical Gaps: {', '.join(no_match['technical_gaps'])}")
            logger.info(f"Suggested Focus: {no_match['suggested_focus']}")
        
        # Save results if not dry run
        if not dry_run:
            logger.info("\nSaving results to Neo4j...")
            # Retry save operation if it fails
            for attempt in range(3):
                try:
                    await analyzer.save_classification_results(use_case['id'], results, db_service.driver)
                    if checkpoint_mgr:
                        checkpoint_mgr.save_checkpoint(use_case['id'])
                    break
                except (ServiceUnavailable, SessionExpired) as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Save attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    await asyncio.sleep(5 * (attempt + 1))
        else:
            logger.info("\nDry run - would make the following changes:")
            if primary:
                logger.info(f"- Create PRIMARY classification relationship:")
                logger.info(f"  UseCase({use_case['id']}) -> AICategory({primary['category']})")
            
            if supporting:
                logger.info(f"\n- Create {len(supporting)} SUPPORTING classification relationship(s)")
            
            if related:
                logger.info(f"\n- Create {len(related)} RELATED classification relationship(s)")
            
            if no_match:
                logger.info("\n- Create NoMatchAnalysis node and relationship")
        
        logger.info("-" * 80)
        
        # Add a small delay between cases to prevent overwhelming the system
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"Error processing case {use_case['id']}: {str(e)}")
        if not checkpoint_mgr:
            raise
        return

async def process_batch_parallel(batch: List[Dict[str, Any]], analyzers: List[LLMAnalyzer], 
                               categories: List[Dict[str, Any]], db_service: Neo4jService, 
                               dry_run: bool, checkpoint_mgr: Optional[CheckpointManager] = None) -> None:
    """Process a batch of use cases in parallel using multiple analyzers."""
    tasks = []
    for i, use_case in enumerate(batch):
        # Round-robin assignment of analyzers
        analyzer = analyzers[i % len(analyzers)]
        tasks.append(process_single_case(use_case, analyzer, categories, db_service, dry_run, checkpoint_mgr))
    
    await asyncio.gather(*tasks)

async def get_no_match_cases(db_service: Neo4jService) -> List[Dict[str, Any]]:
    """Get use cases that either have NoMatchAnalysis or no classifications."""
    query = """
    // Get cases with NoMatchAnalysis or no classifications
    MATCH (u:UseCase)
    WHERE (u)-[:HAS_ANALYSIS]->(:NoMatchAnalysis)
       OR NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
    OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
    WITH u, a
    RETURN {
        use_case: {
            id: u.id,
            name: coalesce(u.name, 'Unnamed Use Case'),
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            status: coalesce(u.status, 'Draft'),
            dev_stage: 'Unknown',
            dev_method: 'Unknown',
            business_terms: [],
            current_category: 'NO MATCH',
            agency: CASE 
                WHEN a IS NOT NULL 
                THEN {
                    name: a.name,
                    abbreviation: COALESCE(a.abbreviation, 'Unknown')
                }
                ELSE {
                    name: 'Unknown Agency',
                    abbreviation: 'Unknown'
                }
            END,
            created_at: toString(datetime()),
            last_updated: toString(datetime())
        }
    } as result
    """
    try:
        results = await db_service.run_query(query)
        # Extract use_case from each result
        return [r["result"]["use_case"] for r in results]
    except Exception as e:
        logger.error(f"Error getting unclassified cases: {str(e)}")
        raise

async def run_classification(num_cases: Optional[int] = 5, dry_run: bool = False, 
                           use_case_id: Optional[str] = None, model: str = "phi4:latest",
                           batch_size: int = 10, checkpoint_file: Optional[str] = None,
                           process_all: bool = False, ollama_url: str = "http://localhost:11434",
                           num_gpus: int = 2, retry_no_match: bool = False):
    """Run LLM classification on use cases."""
    try:
        start_time = datetime.now()
        
        # Initialize services
        db_service = Neo4jService()
        
        # Create multiple analyzers for parallel processing
        analyzers = []
        for i in range(num_gpus):
            analyzer = LLMAnalyzer(model=model, base_url=ollama_url)
            await analyzer.initialize()
            analyzers.append(analyzer)
        
        # Initialize checkpoint manager if file specified
        checkpoint_mgr = CheckpointManager(checkpoint_file) if checkpoint_file else None
        
        logger.info(f"Starting classification test (dry_run: {dry_run}, model: {model}, using {num_gpus} GPUs)")
        
        # Get total use case count
        total_cases = await db_service.get_total_use_cases()
        logger.info(f"Total use cases in database: {total_cases}")
        
        # Get categories
        categories = await db_service.get_all_categories()
        logger.info(f"Loaded {len(categories)} AI technology categories")
        
        # Get use cases based on criteria
        if use_case_id:
            use_case = await get_single_use_case(db_service, use_case_id)
            if not use_case:
                logger.error(f"Use case with ID {use_case_id} not found")
                return
            use_cases = [use_case]
            logger.info(f"Loaded use case {use_case_id} for testing")
        elif retry_no_match:
            use_cases = await get_no_match_cases(db_service)
            logger.info(f"Loaded {len(use_cases)} NO MATCH use cases for retry")
        else:
            # If process_all is True or num_cases exceeds total, process all cases
            limit = None if process_all else num_cases
            use_cases = await db_service.get_test_cases(limit)
            logger.info(f"Loaded {len(use_cases)} use cases for testing")
        
        # Process use cases in batches
        total_processing_time = 0
        for i in range(0, len(use_cases), batch_size):
            batch = use_cases[i:i + batch_size]
            batch_start_time = datetime.now()
            
            logger.info(f"\nProcessing batch {i//batch_size + 1} of {(len(use_cases) + batch_size - 1)//batch_size}")
            await process_batch_parallel(batch, analyzers, categories, db_service, dry_run, checkpoint_mgr)
            
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            total_processing_time += batch_duration
            logger.info(f"Batch processing time: {batch_duration:.2f} seconds")
            
            # Estimate remaining time
            avg_batch_time = total_processing_time / ((i + batch_size) / batch_size)
            remaining_batches = (len(use_cases) - (i + batch_size)) / batch_size
            est_remaining_time = avg_batch_time * remaining_batches
            
            if remaining_batches > 0:
                logger.info(f"Estimated remaining time: {est_remaining_time/60:.1f} minutes")
        
        # Cleanup
        for analyzer in analyzers:
            await analyzer.cleanup()
        await db_service.cleanup()
        
        total_duration = (datetime.now() - start_time).total_seconds()
        avg_case_time = total_processing_time / len(use_cases)
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"Total runtime: {total_duration:.2f} seconds")
        logger.info(f"Average time per case: {avg_case_time:.2f} seconds")
        logger.info(f"Number of cases processed: {len(use_cases)}")
        logger.info(f"GPUs utilized: {num_gpus}")
        logger.info("Classification test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Test LLM classification of federal use cases")
    parser.add_argument("-n", "--num-cases", type=int, default=5,
                      help="Number of use cases to process (default: 5, ignored if --all is used)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Run without saving to database")
    parser.add_argument("--use-case-id", type=str,
                      help="Process a specific use case by ID")
    parser.add_argument("--model", type=str, default="phi4:latest",
                      help="Ollama model to use (default: phi4:latest)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                      help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="Number of cases to process in each batch (default: 10)")
    parser.add_argument("--checkpoint-file", type=str,
                      help="File to store progress (optional)")
    parser.add_argument("--all", action="store_true",
                      help="Process all use cases in the database")
    parser.add_argument("--num-gpus", type=int, default=2,
                      help="Number of GPUs to use for parallel processing (default: 2)")
    parser.add_argument("--retry-no-match", action="store_true",
                      help="Only process cases currently classified as NO MATCH")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_classification(
            args.num_cases, 
            args.dry_run, 
            args.use_case_id, 
            args.model,
            args.batch_size,
            args.checkpoint_file,
            args.all,
            args.ollama_url,
            args.num_gpus,
            args.retry_no_match
        ))
    except KeyboardInterrupt:
        logger.info("Classification interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 