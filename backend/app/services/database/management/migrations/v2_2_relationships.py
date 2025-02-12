import logging
from neo4j import AsyncGraphDatabase
from typing import Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)

async def migrate_v2_2_relationships():
    """
    Migration script to add v2.2 relationships and properties
    """
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )

    try:
        async with driver.session() as session:
            # Add relevance property to HAS_KEYWORD relationships
            query = """
            MATCH ()-[r:HAS_KEYWORD]->()
            WHERE r.relevance IS NULL
            SET r.relevance = 1.0
            RETURN count(r) as updated
            """
            result = await session.run(query)
            record = await result.single()
            logger.info(f"Added relevance property to {record['updated']} HAS_KEYWORD relationships")

            # Add any missing AIClassification nodes
            query = """
            MATCH (u:UseCase)-[r:USES_TECHNOLOGY]->(c:AICategory)
            WHERE NOT (u)-[:CLASSIFIED_AS]->(:AIClassification)
            WITH u, c, r
            CREATE (cl:AIClassification {
                id: toString(randomUUID()),
                match_type: 'PRIMARY',
                confidence: coalesce(r.confidence, 0.8),
                analysis_method: 'LEGACY',
                analysis_version: 'v2.2',
                review_status: 'PENDING',
                classified_at: datetime(),
                classified_by: 'migration',
                last_updated: datetime()
            })
            CREATE (c)-[:CLASSIFIES]->(cl)
            CREATE (u)-[:CLASSIFIED_AS]->(cl)
            DELETE r
            RETURN count(cl) as created
            """
            result = await session.run(query)
            record = await result.single()
            logger.info(f"Migrated {record['created']} legacy classifications to v2.2 schema")

    except Exception as e:
        logger.error(f"Error during v2.2 relationship migration: {str(e)}")
        raise
    finally:
        await driver.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(migrate_v2_2_relationships()) 