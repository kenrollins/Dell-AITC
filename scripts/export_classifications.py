"""
Script to export AI technology classification results from Neo4j to CSV.

Usage:
    python scripts/export_classifications.py [--output OUTPUT_FILE]

Options:
    --output OUTPUT_FILE    Output CSV file path (default: data/output/classification_results_YYYYMMDD_HHMMSS.csv)
"""

import sys
from pathlib import Path
import csv
import json
import logging
from datetime import datetime
import argparse
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.app.services.database.neo4j_service import Neo4jService
from backend.app.config import get_settings

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'export_classifications_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def get_use_case_classifications() -> List[Dict[str, Any]]:
    """Get all use cases with their classifications from Neo4j."""
    db_service = Neo4jService()
    
    try:
        # Updated query to correctly access LLM analysis data
        query = """
        MATCH (u:UseCase)
        OPTIONAL MATCH (u)-[r:CLASSIFIED_AS]->(cat:AICategory)
        OPTIONAL MATCH (u)-[:HAS_ANALYSIS]->(na:NoMatchAnalysis)
        OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
        OPTIONAL MATCH (u)-[:MANAGED_BY]->(b:Bureau)
        RETURN 
            a.name as agency_name,
            a.abbreviation as agency_abbrev,
            b.name as bureau_name,
            u.name as use_case_name,
            u.topic_area as topic_area,
            u.purpose_benefits as purpose_benefits,
            u.outputs as outputs,
            u.stage as stage,
            u.contains_pii as contains_pii,
            u.has_ato as has_ato,
            u.date_initiated as date_initiated,
            u.date_acquisition as date_acquisition,
            u.date_implemented as date_implemented,
            CASE 
                WHEN cat IS NOT NULL THEN cat.name 
                ELSE 'NO MATCH' 
            END as category_name,
            CASE 
                WHEN r IS NOT NULL THEN r.confidence 
                WHEN na IS NOT NULL THEN na.confidence
                ELSE 0.0 
            END as confidence_score,
            CASE 
                WHEN r IS NOT NULL THEN r.reasoning
                WHEN na IS NOT NULL THEN 
                    CASE
                        WHEN na.reason IS NOT NULL THEN na.reason
                        ELSE 'No match - analysis not available'
                    END
                ELSE 'No classification or analysis available'
            END as llm_justification
        ORDER BY a.name, u.name
        """
        
        async with db_service.driver.session() as session:
            result = await session.run(query)
            records = [record.data() async for record in result]
            
        return records
        
    finally:
        await db_service.cleanup()

def format_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Format a Neo4j record for CSV export."""
    # Format dates
    date_fields = [
        'date_initiated', 'date_acquisition', 'date_implemented'
    ]
    for date_field in date_fields:
        if record.get(date_field):
            try:
                record[date_field] = record[date_field].strftime('%Y-%m-%d %H:%M:%S')
            except AttributeError:
                pass
    
    return record

async def export_classifications(output_file: str):
    """Export classification results to CSV."""
    try:
        # Get classification data
        logger.info("Fetching classification data from Neo4j...")
        records = await get_use_case_classifications()
        
        if not records:
            logger.warning("No classification records found")
            return
            
        # Define CSV fields based on the data structure
        fieldnames = [
            'agency_name', 'agency_abbrev', 'bureau_name', 'use_case_name',
            'topic_area', 'purpose_benefits', 'outputs', 'stage',
            'contains_pii', 'has_ato', 'date_initiated', 'date_acquisition',
            'date_implemented', 'category_name', 'confidence_score',
            'llm_justification'
        ]
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        logger.info(f"Writing results to {output_file}...")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                formatted_record = format_record(record)
                writer.writerow({k: formatted_record.get(k, '') for k in fieldnames})
        
        logger.info(f"Successfully exported {len(records)} records to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting classifications: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Export AI technology classification results to CSV")
    
    # Generate default filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"data/output/classification_results_{timestamp}.csv"
    
    parser.add_argument("--output", type=str, 
                       default=default_output,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        import asyncio
        asyncio.run(export_classifications(args.output))
    except KeyboardInterrupt:
        logger.info("Export interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 