#!/usr/bin/env python3
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "kuxFc8HN")  # Use the correct password

def main():
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Count use cases
            result = session.run("MATCH (u:UseCase) RETURN count(u) as count")
            count = result.single()["count"]
            print(f"Total use cases in database: {count}")
            
            # Get sample use cases
            result = session.run("""
                MATCH (u:UseCase)
                RETURN u.name as name
                LIMIT 5
            """)
            print("\nSample use cases:")
            for record in result:
                print(f"- {record['name']}")
                
    finally:
        driver.close()

if __name__ == "__main__":
    main() 