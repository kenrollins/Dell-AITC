from typing import List, Dict, Optional
import logging
from neo4j import AsyncGraphDatabase
from ..config import get_settings
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class AICategory:
    """AI technology category"""
    id: str
    name: str
    definition: str
    keywords: List[str]
    capabilities: List[str]
    business_language: List[str]
    maturity_level: str
    zone: Optional[str] = None
    status: str = 'active'
    version: str = '2.2'

async def load_categories() -> List[AICategory]:
    """Load AI categories from Neo4j
    
    Returns:
        List of AICategory objects
    """
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Get categories with their keywords
            query = """
                MATCH (c:AICategory)
                WHERE c.status = 'active'
                
                // Get technical keywords
                OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k1:Keyword)
                WHERE k1.type = 'technical_keywords'
                WITH c, collect(DISTINCT k1.name) as keywords
                
                // Get capabilities
                OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k2:Keyword)
                WHERE k2.type = 'capabilities'
                WITH c, keywords, collect(DISTINCT k2.name) as capabilities
                
                // Get business terms
                OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k3:Keyword)
                WHERE k3.type = 'business_terms'
                WITH c, keywords, capabilities, collect(DISTINCT k3.name) as business_terms
                
                // Get zone
                OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
                
                RETURN {
                    id: c.id,
                    name: c.name,
                    definition: c.category_definition,
                    keywords: keywords,
                    capabilities: capabilities,
                    business_language: business_terms,
                    maturity_level: c.maturity_level,
                    zone: z.name
                } as category
                """
                
            result = await session.run(query)
            categories = []
            
            async for record in result:
                category = record['category']
                logger.debug(f"Loaded category: {category['name']}")
                
                categories.append(AICategory(
                    id=category.get('id', ''),
                    name=category['name'],
                    definition=category['definition'],
                    keywords=category['keywords'],
                    capabilities=category['capabilities'],
                    business_language=category['business_language'],
                    maturity_level=category.get('maturity_level', 'unknown'),
                    zone=category.get('zone')
                ))
                
            logger.info(f"Successfully loaded {len(categories)} categories from Neo4j")
            return categories
            
    finally:
        await driver.close()

_categories_cache = None

async def get_categories() -> Dict[str, AICategory]:
    """Get AI categories singleton with caching"""
    global _categories_cache
    if _categories_cache is None:
        _categories_cache = await load_categories()
    return _categories_cache 