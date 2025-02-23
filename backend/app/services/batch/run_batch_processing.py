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
from .batch_processor import BatchProcessor, ProcessingPhase
from ...config import get_settings
from ..database.management.verify_database import DatabaseVerifier
from backend.app.utils.logging import setup_logging
import sys
import time

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

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds / 60)
    seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {seconds:.1f}s"
    hours = int(minutes / 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"

def format_metrics(metrics: dict) -> str:
    """Format metrics for display"""
    lines = ["\n=== Processing Metrics ==="]
    
    # Overall stats
    total_time = (metrics['end_time'] - metrics['start_time']).total_seconds()
    lines.append(f"\nTotal Processing Time: {format_duration(total_time)}")
    lines.append(f"Total Cases Processed: {metrics['total_processed']}")
    
    # Phase metrics
    for phase, phase_metrics in metrics.get('phase_metrics', {}).items():
        duration = phase_metrics.get('duration', 0)
        lines.append(f"\n{phase} Phase:")
        lines.append(f"  Duration: {format_duration(duration)}")
        lines.append(f"  Cases Processed: {phase_metrics.get('total_processed', 0)}")
        
        if phase == ProcessingPhase.LOCAL:
            lines.append(f"  High Confidence: {phase_metrics.get('high_confidence', 0)}")
            lines.append(f"  Medium Confidence: {phase_metrics.get('medium_confidence', 0)}")
            lines.append(f"  Low Confidence: {phase_metrics.get('low_confidence', 0)}")
        elif phase == ProcessingPhase.LLM_VERIFY:
            lines.append(f"  Verified Matches: {phase_metrics.get('verified_matches', 0)}")
            lines.append(f"  Downgraded Cases: {phase_metrics.get('downgraded_cases', 0)}")
        elif phase == ProcessingPhase.LLM_FULL:
            lines.append(f"  Matched: {phase_metrics.get('matched', 0)}")
            lines.append(f"  Unmatched: {phase_metrics.get('unmatched', 0)}")
    
    # Error summary
    if metrics.get('errors'):
        lines.append(f"\nTotal Errors: {len(metrics['errors'])}")
        
    return "\n".join(lines)

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run batch processing of use cases")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of use cases to process in each batch"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for classification"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry run mode (no database changes)"
    )
    
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip database verification"
    )
    
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to process"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()

def setup_logging(level: str):
    """Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Reduce verbosity of some loggers
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

async def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing with size {args.batch_size}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be saved to database")
        
    try:
        processor = BatchProcessor(
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            skip_verification=args.skip_verification
        )
        
        await processor.run(max_cases=args.max_cases)
        
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        sys.exit(1)
        
    finally:
        if processor:
            await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 