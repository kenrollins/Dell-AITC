"""
Dell-AITC Batch Processing Runner (v2.2)
Executes batch processing of use cases with monitoring and reporting.

Usage:
    python -m backend.app.services.batch.run_batch_processing [--batch-size N] [--dry-run] [--max-cases N]
"""

import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
import json
from .batch_processor import BatchProcessor
from ...config import get_settings
from ..database.management.verify_database import DatabaseVerifier

# Configure logging
log_dir = Path("logs/batch_processing")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'batch_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def verify_database():
    """Verify database state before processing."""
    settings = get_settings()
    verifier = DatabaseVerifier(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password
    )
    
    try:
        # Run verifications
        logger.info("Verifying database state...")
        node_counts = verifier.check_node_counts()
        relationship_counts = verifier.check_relationships()
        integrity_results = verifier.check_data_integrity()
        
        # Log results
        logger.info("\n=== Database Verification Results ===")
        
        logger.info("\nNode Counts:")
        for result in node_counts:
            logger.info(f"{result['label']}: {result['count']} [{result['status']}]")
            
        logger.info("\nRelationship Counts:")
        for result in relationship_counts:
            logger.info(f"{result['relationship']}: {result['count']} [{result['status']}]")
            
        logger.info("\nData Integrity:")
        violations = sum(r['violations'] for r in integrity_results)
        if violations > 0:
            logger.warning(f"Found {violations} data integrity violations")
            return False
            
        logger.info("Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False
    finally:
        verifier.close()

async def save_metrics(metrics: dict, batch_size: int):
    """Save processing metrics to file."""
    output_dir = Path("data/output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir / f'batch_metrics_{timestamp}.json'
    
    # Add batch configuration
    metrics['configuration'] = {
        'batch_size': batch_size,
        'timestamp': timestamp
    }
    
    # Convert datetime objects to strings
    metrics['start_time'] = metrics['start_time'].isoformat() if metrics['start_time'] else None
    metrics['end_time'] = metrics['end_time'].isoformat() if metrics['end_time'] else None
    for error in metrics['errors']:
        error['timestamp'] = error['timestamp'].isoformat()
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_file}")

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run batch processing of use cases")
    parser.add_argument('--batch-size', type=int, default=20,
                       help="Number of use cases to process in each batch")
    parser.add_argument('--dry-run', action='store_true',
                       help="Run without saving to database")
    parser.add_argument('--min-confidence', type=float, default=0.45,
                       help="Minimum confidence threshold for classification")
    parser.add_argument('--max-cases', type=int, default=None,
                       help="Maximum number of cases to process (default: all)")
    args = parser.parse_args()
    
    try:
        # Verify database first
        if not await verify_database():
            logger.error("Database verification failed. Aborting batch processing.")
            return
            
        # Initialize batch processor
        processor = BatchProcessor(
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run
        )
        await processor.initialize()
        
        try:
            # Process batches until max_cases is reached or no more cases
            logger.info(f"Starting batch processing with size {args.batch_size}")
            total_processed = 0
            metrics = None
            
            while True:
                if args.max_cases and total_processed >= args.max_cases:
                    logger.info(f"Reached maximum cases limit ({args.max_cases})")
                    break
                    
                # Adjust batch size for last batch if max_cases specified
                if args.max_cases:
                    remaining = args.max_cases - total_processed
                    if remaining < args.batch_size:
                        processor.batch_size = remaining
                
                batch_metrics = await processor.process_batch()
                if not batch_metrics or batch_metrics['total_processed'] == 0:
                    logger.info("No more cases to process")
                    break
                    
                total_processed += batch_metrics['total_processed']
                metrics = batch_metrics  # Keep last batch metrics
                
                logger.info(f"Processed {total_processed} cases total")
            
            # Save final metrics
            if metrics:
                await save_metrics(metrics, args.batch_size)
            
        finally:
            # Ensure cleanup
            await processor.cleanup()
            
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 