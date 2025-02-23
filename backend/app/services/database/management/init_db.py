"""Database initialization module."""
import logging
from typing import Optional

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError

from ....config import get_settings

logger = logging.getLogger(__name__)

async def init_db(driver: Optional[AsyncGraphDatabase] = None) -> None:
    """Initialize the database with required constraints and indexes."""
    if not driver:
        settings = get_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )

    try:
        async with driver.session() as session:
            # Create constraints
            await session.run("""
                CREATE CONSTRAINT use_case_id IF NOT EXISTS 
                FOR (u:UseCase) REQUIRE u.id IS UNIQUE
            """)
            
            await session.run("""
                CREATE CONSTRAINT ai_category_name IF NOT EXISTS
                FOR (c:AICategory) REQUIRE c.name IS UNIQUE
            """)

            # Create indexes
            await session.run("""
                CREATE INDEX use_case_name IF NOT EXISTS
                FOR (u:UseCase) ON (u.name)
            """)

            # Ensure CLASSIFIED_AS relationship type exists by creating a temporary relationship
            await session.run("""
                MERGE (u:UseCase {id: 'temp'})
                MERGE (c:AICategory {name: 'temp'})
                MERGE (u)-[r:CLASSIFIED_AS]->(c)
                WITH u, c, r
                DELETE r, u, c
            """)

            logger.info("Database initialized successfully")
    except ClientError as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if not driver:
            await driver.close() 