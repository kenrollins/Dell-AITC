"""
Quick script to check for NO MATCH classifications and NoMatchAnalysis nodes in the database.
"""

import asyncio
from backend.app.services.database.neo4j_service import Neo4jService

async def check_no_matches():
    db = Neo4jService()
    try:
        # Count NO MATCH classifications
        query = '''
        MATCH (u:UseCase)-[r:CLASSIFIED_AS]->(c:AICategory)
        WHERE c.name = 'NO MATCH'
        RETURN count(u) as count
        '''
        result = await db.run_query(query)
        print(f'Number of NO MATCH classifications: {result[0]["count"]}')
        
        # Count NoMatchAnalysis nodes
        query = '''
        MATCH (u:UseCase)-[r:HAS_ANALYSIS]->(n:NoMatchAnalysis)
        RETURN count(u) as count
        '''
        result = await db.run_query(query)
        print(f'Number of cases with NoMatchAnalysis: {result[0]["count"]}')
        
        # Count cases with no classifications
        query = '''
        MATCH (u:UseCase)
        WHERE NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
        RETURN count(u) as count
        '''
        result = await db.run_query(query)
        print(f'Number of cases with no classifications: {result[0]["count"]}')
        
        # Count overlap (cases that have both NoMatchAnalysis and no classifications)
        query = '''
        MATCH (u:UseCase)-[:HAS_ANALYSIS]->(:NoMatchAnalysis)
        WHERE NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
        RETURN count(u) as count
        '''
        result = await db.run_query(query)
        print(f'Number of cases with both NoMatchAnalysis and no classifications: {result[0]["count"]}')
        
        # Count total unique cases to retry
        query = '''
        MATCH (u:UseCase)
        WHERE (u)-[:HAS_ANALYSIS]->(:NoMatchAnalysis)
           OR NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
        RETURN count(u) as count
        '''
        result = await db.run_query(query)
        print(f'\nTotal unique cases to retry: {result[0]["count"]}')
        
        # Get sample of these cases
        query = '''
        MATCH (u:UseCase)
        WHERE (u)-[:HAS_ANALYSIS]->(:NoMatchAnalysis)
           OR NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
        OPTIONAL MATCH (u)-[:HAS_ANALYSIS]->(n:NoMatchAnalysis)
        RETURN u.name as name, u.id as id,
               CASE WHEN n IS NOT NULL THEN n.reason ELSE 'No previous analysis' END as reason,
               CASE 
                 WHEN (u)-[:HAS_ANALYSIS]->(:NoMatchAnalysis) AND NOT (u)-[:CLASSIFIED_AS]->(:AICategory)
                 THEN 'Both'
                 WHEN (u)-[:HAS_ANALYSIS]->(:NoMatchAnalysis)
                 THEN 'Has NoMatchAnalysis'
                 ELSE 'No Classifications'
               END as type
        LIMIT 5
        '''
        result = await db.run_query(query)
        if result:
            print("\nSample cases to retry:")
            for record in result:
                print(f"\n- {record['name']} (ID: {record['id']})")
                print(f"  Type: {record['type']}")
                print(f"  Reason: {record['reason']}")
        
    finally:
        await db.cleanup()

if __name__ == "__main__":
    asyncio.run(check_no_matches()) 