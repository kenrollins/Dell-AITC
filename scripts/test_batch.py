"""
Simple test script to run batch processing with reduced output and error recovery
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import json
import asyncio
from typing import Dict, Any, Union
import pprint
from neo4j.graph import Node
import copy

# Configure logging
log_dir = Path('logs/batch_test')
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'batch_test_{timestamp}.log'

# Configure file handler with detailed logging
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
))

# Configure console handler with minimal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Setup root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)

# Reduce logging from other modules
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('neo4j').setLevel(logging.WARNING)

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from backend.app.services.batch.batch_processor import BatchProcessor
from backend.app.services.database.management.verify_database import DatabaseVerifier
from backend.app.config import get_settings

def serialize_node(obj: Any) -> Any:
    """Convert Neo4j Node objects to dictionaries for JSON serialization."""
    try:
        if isinstance(obj, Node):
            # Only include essential properties to avoid circular references
            return {
                'id': obj.id,
                'labels': list(obj.labels),
                'properties': dict(obj)
            }
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # For custom objects, only include their dictionary representation
            return dict(obj.__dict__)
        return obj
    except Exception as e:
        logging.warning(f"Error serializing object {type(obj)}: {str(e)}")
        return str(obj)

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f'checkpoint_{timestamp}.json'
        self.metrics = {}
        
    def save_checkpoint(self, metrics: Dict[str, Any]) -> None:
        """Save processing metrics to checkpoint file."""
        try:
            # Deep copy metrics to avoid modifying the original
            metrics_copy = copy.deepcopy(metrics)
            
            # Remove any potential circular references
            if 'case_results' in metrics_copy:
                for case in metrics_copy['case_results']:
                    if 'use_case' in case:
                        case['use_case'] = serialize_node(case['use_case'])
                    if 'category' in case:
                        case['category'] = serialize_node(case['category'])
            
            # Convert to JSON-serializable format
            json_str = json.dumps(metrics_copy, default=serialize_node, indent=2)
            
            with open(self.checkpoint_file, 'w') as f:
                f.write(json_str)
                
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load metrics from checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}

def log_no_match_details(use_case: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Log detailed information about no-match cases."""
    logger.debug("\n=== No Match Case Analysis ===")
    logger.debug(f"Use Case ID: {use_case.get('id')}")
    logger.debug(f"Use Case Name: {use_case.get('name')}")
    
    # Log scores
    logger.debug("\nConfidence Scores:")
    logger.debug(f"  Keyword Score: {results.get('keyword_score', 0):.3f}")
    logger.debug(f"  Semantic Score: {results.get('semantic_score', 0):.3f}")
    logger.debug(f"  LLM Score: {results.get('llm_score', 0):.3f}")
    logger.debug(f"  Overall Confidence: {results.get('confidence', 0):.3f}")
    
    # Log matched keywords if any
    if results.get('matched_keywords'):
        logger.debug("\nPartial Keyword Matches:")
        logger.debug(f"  {results['matched_keywords']}")
    
    # Log field match details
    if results.get('match_details'):
        logger.debug("\nField Match Details:")
        logger.debug(pprint.pformat(results['match_details'], indent=2))
    
    # Log LLM analysis if available
    if results.get('llm_explanation'):
        logger.debug("\nLLM Analysis:")
        logger.debug(f"  Explanation: {results['llm_explanation']}")
        
        if results.get('field_match_scores'):
            logger.debug("\nField Match Scores:")
            logger.debug(pprint.pformat(results['field_match_scores'], indent=2))
            
        if results.get('matched_terms'):
            logger.debug("\nMatched Terms:")
            logger.debug(pprint.pformat(results['matched_terms'], indent=2))
    
    # Log alternative matches if any
    if results.get('alternative_matches'):
        logger.debug("\nAlternative Matches Considered:")
        for alt in results['alternative_matches']:
            logger.debug(f"  Category: {alt.get('category_name')}")
            logger.debug(f"  Confidence: {alt.get('confidence', 0):.3f}")
            logger.debug(f"  Reasoning: {alt.get('reasoning', '')}")
    
    logger.debug("\nUse Case Content:")
    for field in ['description', 'purpose_benefits', 'outputs', 'topic_area', 'stage']:
        if use_case.get(field):
            logger.debug(f"  {field}: {use_case[field]}")
    
    logger.debug("=" * 80)

async def verify_database_state() -> bool:
    """Verify database state before processing."""
    settings = get_settings()
    verifier = DatabaseVerifier(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password
    )
    
    try:
        print("\nVerifying database state...")
        
        # Check node counts
        node_counts = verifier.check_node_counts()
        relationship_counts = verifier.check_relationships()
        
        # Log results
        print("\nNode Counts:")
        all_nodes_ok = True
        for result in node_counts:
            status = "✓" if result['status'] == 'OK' else "×"
            print(f"  {status} {result['label']}: {result['count']}")
            if result['status'] != 'OK':
                all_nodes_ok = False
                
        print("\nRelationship Counts:")
        all_rels_ok = True
        for result in relationship_counts:
            status = "✓" if result['status'] == 'OK' else "×"
            print(f"  {status} {result['relationship']}: {result['count']}")
            if result['status'] != 'OK':
                all_rels_ok = False
                
        if not (all_nodes_ok and all_rels_ok):
            print("\n⚠️  Warning: Some database components are missing or empty")
            proceed = input("Do you want to proceed anyway? (y/n): ").lower()
            return proceed == 'y'
            
        print("\n✓ Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False
    finally:
        verifier.close()

async def main():
    """Run batch processing test with error recovery."""
    checkpoint_dir = Path('data/checkpoints')
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    try:
        # Initialize processor with smaller batch size and increased timeout
        processor = BatchProcessor(
            batch_size=5,  # Process 5 use cases
            min_confidence=0.45,
            dry_run=True,  # Test mode - no database changes
            api_timeout=60.0  # Increase API timeout to 60 seconds
        )
        
        # Initialize connections
        await processor.initialize()
        logger.debug("Processor initialized successfully")
        
        # Process batch with progress bar
        print("\nProcessing use cases:")
        with tqdm(total=5, desc="Progress", unit="cases", ncols=80) as pbar:
            metrics = await processor.process_batch()
            
            # Add delay between API calls (500ms)
            await asyncio.sleep(0.5)
            
            # Log no-match cases in detail
            for case_result in metrics.get('case_results', []):
                if case_result['match_type'] == 'NONE' or case_result.get('confidence', 0) < 0.45:
                    log_no_match_details(case_result['use_case'], case_result)
                    # Add delay after each detailed analysis
                    await asyncio.sleep(0.2)
            
            pbar.update(metrics['total_processed'])
            
            # Save checkpoint after each case
            checkpoint_manager.save_checkpoint(metrics)
            
            # Print interim summary
            print(f"\nProcessed {metrics['total_processed']} cases:")
            print(f"- High confidence matches: {metrics['high_confidence_matches']}")
            print(f"- Low confidence matches: {metrics['low_confidence_matches']}")
            print(f"- No matches: {metrics['no_matches']}")
            print(f"Average processing time: {metrics['avg_processing_time']:.2f}s per case")
        
        # Log detailed metrics to file
        logger.debug("Detailed Processing Metrics:")
        for key, value in metrics.items():
            if key != 'case_results':  # Skip detailed case results in this summary
                logger.debug(f"{key}: {value}")
        
        # Calculate confidence distribution
        total = metrics['total_processed']
        high_conf = metrics['high_confidence_matches']
        low_conf = metrics['low_confidence_matches']
        no_match = metrics['no_matches']
        
        # Display enhanced summary
        print("\nProcessing Summary:")
        print(f"Total Processed: {total}")
        print(f"\nConfidence Distribution:")
        print(f"  ✓ High Confidence ({high_conf/total*100:.1f}%): {high_conf}")
        print(f"  ○ Low Confidence ({low_conf/total*100:.1f}%): {low_conf}")
        print(f"  × No Matches    ({no_match/total*100:.1f}%): {no_match}")
        
        if metrics['processing_times']:
            avg_time = metrics['avg_processing_time']
            print(f"\nPerformance Metrics:")
            print(f"  Average Time: {avg_time:.2f}s per case")
            print(f"  Fastest Case: {min(metrics['processing_times']):.2f}s")
            print(f"  Slowest Case: {max(metrics['processing_times']):.2f}s")
            print(f"  Estimated for 2000: {(avg_time * 2000 / 60):.1f} minutes")
        
        if metrics['errors']:
            print(f"\n⚠️  Errors: {len(metrics['errors'])}")
            print("Check log file for details:", log_file)
        
        print("\nNote: This was a dry run - no changes were made to the database")
        print(f"Detailed logs available at: {log_file}")
        print(f"Checkpoint saved at: {checkpoint_manager.checkpoint_file}")
    
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}", exc_info=True)
        print(f"\n× Error: {str(e)}")
        print("Check log file for details:", log_file)
        
        # Try to load last checkpoint
        last_checkpoint = checkpoint_manager.load_checkpoint()
        if last_checkpoint:
            print("\nRecovery checkpoint available with partial results.")
        raise
    finally:
        # Cleanup
        await processor.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        sys.exit(1) 