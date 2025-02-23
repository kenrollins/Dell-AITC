from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase
from ...config import get_settings

class Neo4jService:
    def __init__(self):
        """Initialize Neo4j service with connection settings."""
        self.settings = get_settings()
        self.driver = AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password)
        )

    async def run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Run a Cypher query and return results."""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return [record.data() async for record in result]

    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """Fetch all technology categories with their properties"""
        query = """
        MATCH (c:AICategory)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
        WITH c, z, 
             collect(DISTINCT {
                name: k.name,
                type: coalesce(k.type, 'technical'),
                relevance_score: coalesce(k.relevance_score, 0.5)
             }) as keywords
        RETURN {
            id: c.id,
            name: coalesce(c.name, 'Unnamed Category'),
            category_definition: c.category_definition,
            maturity_level: coalesce(c.maturity_level, 'Unknown'),
            status: coalesce(c.status, 'Active'),
            zone: coalesce(z.name, 'Unassigned'),
            keywords: [kw IN keywords WHERE kw.type = 'technical'],
            business_terms: [kw IN keywords WHERE kw.type = 'business'],
            created_at: toString(datetime()),
            last_updated: toString(datetime()),
            version: coalesce(c.version, '1.0')
        } as category
        ORDER BY category.name
        """
        categories = await self.run_query(query)
        return [record['category'] for record in categories]

    async def get_total_use_cases(self) -> int:
        """Get total count of use cases in the database."""
        query = """
        MATCH (u:UseCase)
        RETURN count(u) as total
        """
        result = await self.run_query(query)
        return result[0]['total']

    async def get_test_cases(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get use cases for testing.
        
        Args:
            limit: Optional limit on number of cases. If None, returns all cases.
        """
        query = """
        MATCH (u:UseCase)
        OPTIONAL MATCH (u)-[:BELONGS_TO]->(c:AICategory)
        OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
        RETURN {
            id: u.id,
            name: coalesce(u.name, 'Unnamed Use Case'),
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            status: coalesce(u.status, 'Draft'),
            dev_stage: 'Unknown',
            dev_method: 'Unknown',
            business_terms: [],
            current_category: c.name,
            agency: CASE 
                WHEN a IS NOT NULL 
                THEN {
                    name: a.name,
                    abbreviation: COALESCE(a.abbreviation, 'Unknown')
                }
                ELSE {
                    name: 'Unknown Agency',
                    abbreviation: 'Unknown'
                }
            END,
            created_at: toString(datetime()),
            last_updated: toString(datetime())
        } as use_case
        """ + (f" LIMIT {limit}" if limit is not None else "")
        
        cases = await self.run_query(query)
        return [record['use_case'] for record in cases]

    async def cleanup(self):
        """Close the Neo4j driver connection."""
        await self.driver.close()

    async def get_category_details(self, category_id: str) -> Dict:
        """Get detailed information about a specific AI category."""
        query = """
        MATCH (c:AICategory {id: $category_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
        WITH c, z, 
             collect(DISTINCT {
                name: k.name,
                type: coalesce(k.type, 'technical'),
                relevance_score: coalesce(k.relevance_score, 0.5)
            }) as keywords
        RETURN {
            id: c.id,
            name: coalesce(c.name, 'Unnamed Category'),
            category_definition: c.category_definition,
            maturity_level: coalesce(c.maturity_level, 'Unknown'),
            status: coalesce(c.status, 'Active'),
            zone: coalesce(z.name, 'Unassigned'),
            keywords: [kw IN keywords WHERE kw.type = 'technical'],
            business_terms: [kw IN keywords WHERE kw.type = 'business'],
            created_at: toString(datetime()),
            last_updated: toString(datetime()),
            version: coalesce(c.version, '1.0')
        } as category
        """
        category = await self.run_query(query, {"category_id": category_id})
        return category[0]['category'] 