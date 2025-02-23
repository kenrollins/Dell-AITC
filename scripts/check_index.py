"""
Check if agency_abbreviation index exists in Neo4j database.
This script only performs read operations - no modifications.

Usage:
    python scripts/check_index.py
"""

import sys
from pathlib import Path
import logging

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / 'backend'
sys.path.append(str(backend_dir))

from app.services.database.neo4j_service import Neo4jService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_index():
    """Check if agency_abbreviation index exists."""
    try:
        service = Neo4jService()
        
        # List all indexes
        logger.info("Checking database indexes...")
        result = await service.run_query("""
        SHOW INDEXES
        YIELD name, type
        RETURN name, type
        ORDER BY name
        """)
        
        # Check for agency_abbreviation
        agency_index = None
        for record in result:
            if 'agency_abbreviation' in record['name'].lower():
                agency_index = record
                break
                
        if agency_index:
            logger.info("\nFound agency_abbreviation index:")
            logger.info(f"Name: {agency_index['name']}")
            logger.info(f"Type: {agency_index['type']}")
        else:
            logger.info("\nNo agency_abbreviation index found in database")
            logger.info("\nExisting indexes:")
            for idx in result:
                logger.info(f"- {idx['name']} ({idx['type']}")
                
        await service.cleanup()
        return True
            
    except Exception as e:
        logger.error(f"Error checking index: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(check_index())
    exit(0 if success else 1) 