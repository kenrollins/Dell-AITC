from typing import List, Dict, Optional
import logging
from neo4j import AsyncGraphDatabase
from ..config import get_settings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AICategory:
    def __init__(
        self,
        id: str,
        name: str,
        category_definition: str,
        status: str = "Active",
        maturity_level: Optional[str] = None,
        zone_id: Optional[str] = None,
        keywords: List[str] = None,
        capabilities: List[str] = None,
        business_language: List[str] = None,
        version: Optional[str] = None
    ):
        self.id = id
        self.name = name
        self.definition = category_definition
        self.status = status
        self.maturity_level = maturity_level
        self.zone_id = zone_id
        self.keywords = [k.lower().strip() for k in (keywords or [])]
        self.capabilities = [c.lower().strip() for c in (capabilities or [])]
        self.business_language = [b.lower().strip() for b in (business_language or [])]
        self.version = version

    @staticmethod
    async def load_categories() -> Dict[str, 'AICategory']:
        """Load AI categories from Neo4j database"""
        settings = get_settings()
        categories = {}
        
        # Initialize Neo4j driver
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        try:
            async with driver.session() as session:
                # Query to get categories with their keywords and capabilities
                query = """
                MATCH (c:AICategory)
                WHERE c.status = 'active'
                OPTIONAL MATCH (c)-[r:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
                WITH c, z,
                     collect(DISTINCT {
                         name: k.name,
                         relevance: r.relevance
                     }) as keywords
                RETURN {
                    id: c.id,
                    name: c.name,
                    definition: c.category_definition,
                    status: c.status,
                    maturity_level: c.maturity_level,
                    zone_id: z.id,
                    zone: z.name,
                    keywords: [k in keywords | k.name],
                    version: c.version
                } as category
                """
                
                result = await session.run(query)
                
                async for record in result:
                    cat = record["category"]
                    try:
                        categories[cat["name"]] = AICategory(
                            id=cat["id"],
                            name=cat["name"],
                            category_definition=cat.get("definition"),
                            status=cat.get("status", "active"),
                            maturity_level=cat.get("maturity_level"),
                            zone_id=cat.get("zone_id"),
                            keywords=cat.get("keywords", []),
                            capabilities=[],  # Will be loaded in next query
                            business_language=[],  # Will be loaded in next query
                            version=cat.get("version")
                        )
                        logger.debug(f"Loaded category: {cat['name']}")
                    except Exception as e:
                        logger.error(f"Error processing category {cat.get('name', 'unknown')}: {str(e)}")
                        continue
                
                # Load capabilities and business terms in a separate query
                if categories:
                    query = """
                    MATCH (c:AICategory)
                    WHERE c.name IN $category_names
                    OPTIONAL MATCH (c)-[r1:HAS_KEYWORD]->(k:Keyword)
                    WHERE r1.type = 'capability'
                    WITH c, collect(DISTINCT k.name) as capabilities
                    OPTIONAL MATCH (c)-[r2:HAS_KEYWORD]->(k:Keyword)
                    WHERE r2.type = 'business_term'
                    RETURN 
                        c.name as name,
                        capabilities,
                        collect(DISTINCT k.name) as business_terms
                    """
                    
                    result = await session.run(query, 
                                            category_names=list(categories.keys()))
                    
                    async for record in result:
                        if record["name"] in categories:
                            cat = categories[record["name"]]
                            # Handle capabilities and business terms
                            capabilities = record["capabilities"] or []
                            cat.capabilities = [c.lower().strip() for c in capabilities]
                            
                            business_terms = record["business_terms"] or []
                            cat.business_language = [b.lower().strip() for b in business_terms]
                
                logger.info(f"Successfully loaded {len(categories)} categories from Neo4j")
                return categories
                
        except Exception as e:
            logger.error(f"Error loading categories from Neo4j: {str(e)}")
            raise
        finally:
            await driver.close()

_categories_cache = None

async def get_categories() -> Dict[str, AICategory]:
    """Get AI categories singleton with caching"""
    global _categories_cache
    if _categories_cache is None:
        _categories_cache = await AICategory.load_categories()
    return _categories_cache 