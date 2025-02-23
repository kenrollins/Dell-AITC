import logging
from typing import Dict, List, Optional
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

class DatabaseVerifier:
    """Verifies database operations and data consistency"""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize database verifier
        
        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
    async def initialize(self):
        """Initialize database connection"""
        logger.info("Initializing database verifier")
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        
    async def cleanup(self):
        """Clean up resources"""
        if self.driver:
            await self.driver.close()
            
    async def verify_category_exists(self, category_name: str) -> bool:
        """Verify that a category exists in the database
        
        Args:
            category_name: Name of category to verify
            
        Returns:
            True if category exists, False otherwise
        """
        if not self.driver:
            await self.initialize()
            
        async with self.driver.session() as session:
            query = """
                MATCH (c:AICategory {name: $name})
                RETURN count(c) as count
                """
            result = await session.run(query, name=category_name)
            record = await result.single()
            return record and record["count"] > 0
            
    async def verify_use_case_exists(self, use_case_id: str) -> bool:
        """Verify that a use case exists in the database
        
        Args:
            use_case_id: ID of use case to verify
            
        Returns:
            True if use case exists, False otherwise
        """
        if not self.driver:
            await self.initialize()
            
        async with self.driver.session() as session:
            query = """
                MATCH (u:UseCase {id: $id})
                RETURN count(u) as count
                """
            result = await session.run(query, id=use_case_id)
            record = await result.single()
            return record and record["count"] > 0
            
    async def verify_relationship_exists(
        self,
        use_case_id: str,
        category_name: str,
        relationship_type: str = "CLASSIFIED_AS"
    ) -> bool:
        """Verify that a relationship exists between a use case and category
        
        Args:
            use_case_id: ID of use case
            category_name: Name of category
            relationship_type: Type of relationship to verify
            
        Returns:
            True if relationship exists, False otherwise
        """
        if not self.driver:
            await self.initialize()
            
        async with self.driver.session() as session:
            query = """
                MATCH (u:UseCase {id: $use_case_id})
                -[r:$relationship_type]->
                (c:AICategory {name: $category_name})
                RETURN count(r) as count
                """
            result = await session.run(
                query,
                use_case_id=use_case_id,
                category_name=category_name,
                relationship_type=relationship_type
            )
            record = await result.single()
            return record and record["count"] > 0
            
    async def save_classification(
        self,
        use_case_id: str,
        category_name: str,
        confidence: float,
        method: str
    ) -> bool:
        """Save a classification result to the database
        
        Args:
            use_case_id: ID of use case
            category_name: Name of category
            confidence: Confidence score
            method: Classification method used
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.driver:
            await self.initialize()
            
        async with self.driver.session() as session:
            query = """
                MATCH (u:UseCase {id: $use_case_id})
                MATCH (c:AICategory {name: $category_name})
                MERGE (u)-[r:CLASSIFIED_AS]->(c)
                SET r.confidence = $confidence,
                    r.method = $method,
                    r.classified_at = datetime()
                RETURN r
                """
            try:
                result = await session.run(
                    query,
                    use_case_id=use_case_id,
                    category_name=category_name,
                    confidence=confidence,
                    method=method
                )
                record = await result.single()
                return record is not None
            except Exception as e:
                self.logger.error(f"Error saving classification: {str(e)}")
                return False 