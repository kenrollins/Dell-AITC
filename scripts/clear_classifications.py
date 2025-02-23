"""
Script to clear all existing classifications from the Neo4j database.

Usage:
    python scripts/clear_classifications.py
"""

import sys
from pathlib import Path
import asyncio
from typing import Dict, Any
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.app.services.database.neo4j_service import Neo4jService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def clear_classifications():
    """Clear all existing classifications from the database."""
    try:
        db_service = Neo4jService()
        
        # Delete all CLASSIFIED_AS relationships
        query1 = """
        MATCH ()-[r:CLASSIFIED_AS]->()
        DELETE r
        """
        
        # Delete all NoMatchAnalysis nodes and their relationships
        query2 = """
        MATCH (n:NoMatchAnalysis)
        DETACH DELETE n
        """
        
        logger.info("Clearing all existing classifications...")
        await db_service.run_query(query1)
        logger.info("Cleared CLASSIFIED_AS relationships")
        
        await db_service.run_query(query2)
        logger.info("Cleared NoMatchAnalysis nodes")
        
        await db_service.cleanup()
        logger.info("Classification cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error clearing classifications: {str(e)}")
        raise

def main():
    try:
        asyncio.run(clear_classifications())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 