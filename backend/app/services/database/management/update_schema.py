"""
Dell-AITC Schema Update Script (v2.2)
Updates Neo4j schema with missing properties and indexes.

Usage:
    python -m backend.app.services.database.management.update_schema
"""

import logging
from neo4j import AsyncGraphDatabase
from ....config import get_settings

logger = logging.getLogger(__name__)

async def update_schema():
    """Update the Neo4j schema with missing properties and indexes."""
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Update Keyword nodes to include relevance_score
            logger.info("Updating Keyword nodes with relevance_score...")
            result = await session.run("""
                MATCH (k:Keyword)
                WHERE k.relevance_score IS NULL
                SET k.relevance_score = 0.5
                RETURN count(k) as updated
            """)
            record = await result.single()
            logger.info(f"Updated {record['updated']} Keyword nodes")
            
            # Update HAS_KEYWORD relationships to include relevance
            logger.info("Updating HAS_KEYWORD relationships with relevance...")
            result = await session.run("""
                MATCH (c:AICategory)-[r:HAS_KEYWORD]->(k:Keyword)
                WHERE r.relevance IS NULL
                SET r.relevance = 0.5
                RETURN count(r) as updated
            """)
            record = await result.single()
            logger.info(f"Updated {record['updated']} HAS_KEYWORD relationships")
            
            # Create index for relevance_score
            logger.info("Creating index for Keyword relevance_score...")
            await session.run("""
                CREATE INDEX keyword_relevance IF NOT EXISTS
                FOR (k:Keyword) ON (k.relevance_score)
            """)
            
            # Create index for relationship relevance
            logger.info("Creating index for HAS_KEYWORD relationship relevance...")
            await session.run("""
                CREATE INDEX has_keyword_relevance IF NOT EXISTS
                FOR ()-[r:HAS_KEYWORD]-() ON (r.relevance)
            """)
            
            logger.info("Schema update completed successfully")
            
    except Exception as e:
        logger.error(f"Error updating schema: {str(e)}")
        raise
    finally:
        await driver.close()

async def main():
    """Main entry point."""
    try:
        await update_schema()
    except Exception as e:
        logger.error(f"Schema update failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 