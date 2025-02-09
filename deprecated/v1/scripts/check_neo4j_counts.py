#!/usr/bin/env python3
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        print("Error: NEO4J_PASSWORD environment variable not set")
        return
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()
    
    try:
        # Get counts for each node type
        result = session.run("""
        MATCH (n) 
        RETURN DISTINCT labels(n) as NodeType, count(*) as Count 
        ORDER BY Count DESC
        """)
        
        print("\nNode counts in Neo4j:")
        print("----------------------")
        for record in result:
            print(f"{record['NodeType']}: {record['Count']}")
            
        # Get relationship counts
        result = session.run("""
        MATCH ()-[r]->() 
        RETURN DISTINCT type(r) as RelType, count(*) as Count 
        ORDER BY Count DESC
        """)
        
        print("\nRelationship counts in Neo4j:")
        print("-----------------------------")
        for record in result:
            print(f"{record['RelType']}: {record['Count']}")
            
    finally:
        session.close()
        driver.close()

if __name__ == "__main__":
    main() 