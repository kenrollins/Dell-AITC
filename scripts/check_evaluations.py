from neo4j import GraphDatabase
import os

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    # Check CategoryEvaluation nodes
    result = session.run('MATCH (n:CategoryEvaluation) RETURN count(n) as count')
    print(f'CategoryEvaluation nodes: {result.single()["count"]}')
    
    # Check HAS_EVALUATION relationships
    result = session.run('MATCH ()-[r:HAS_EVALUATION]->() RETURN count(r) as count')
    print(f'HAS_EVALUATION relationships: {result.single()["count"]}')

driver.close() 