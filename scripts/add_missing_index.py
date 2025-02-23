"""
Add missing agency_abbreviation index to Neo4j database.

Usage:
    python scripts/add_missing_index.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

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

async def add_missing_index():
    """Add the missing agency_abbreviation index."""
    try:
        service = Neo4jService()
        
        # First check if index exists
        result = await service.run_query("""
        SHOW INDEXES
        YELD name
        WHERE name CONTAINS 'agency_abbreviation'
        RETURN count(*) as count
        """)
        
        if result[0]['count'] > 0:
            logger.info("Index 'agency_abbreviation' already exists")
            return True
            
        # Create the index
        logger.info("Creating agency_abbreviation index...")
        await service.run_query("""
        CREATE INDEX agency_abbreviation IF NOT EXISTS 
        FOR (a:Agency) ON (a.abbreviation)
        """)
        
        # Verify creation
        result = await service.run_query("""
        SHOW INDEXES
        YIELD name
        WHERE name CONTAINS 'agency_abbreviation'
        RETURN count(*) as count
        """)
        
        if result[0]['count'] > 0:
            logger.info("Successfully created agency_abbreviation index")
            return True
        else:
            logger.error("Failed to create index")
            return False
            
    except Exception as e:
        logger.error(f"Error adding index: {str(e)}")
        return False
    finally:
        await service.cleanup()

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(add_missing_index())
    exit(0 if success else 1) 