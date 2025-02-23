import asyncio
from neo4j import AsyncGraphDatabase
from backend.app.config import get_settings

async def check_keywords():
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Check total count
            result = await session.run('MATCH (k:Keyword) RETURN count(k) as count')
            record = await result.single()
            print(f'\nTotal number of keywords: {record["count"]}')
            
            # Check keywords by type
            result = await session.run('''
                MATCH (k:Keyword)
                WITH k.type as type, collect(k.name) as keywords
                RETURN type, keywords
                ORDER BY type
            ''')
            
            print("\nKeywords by type:")
            async for record in result:
                keyword_type = record["type"] or "no_type"
                keywords = record["keywords"]
                print(f"\n{keyword_type}:")
                for kw in keywords:
                    print(f"- {kw}")
                    
            # Check relationships
            result = await session.run('''
                MATCH (c:AICategory)-[r:HAS_KEYWORD]->(k:Keyword)
                RETURN c.name as category, count(r) as rel_count
                ORDER BY rel_count DESC
            ''')
            
            print("\nKeyword relationships by category:")
            async for record in result:
                print(f"{record['category']}: {record['rel_count']} keywords")
                
    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(check_keywords()) 