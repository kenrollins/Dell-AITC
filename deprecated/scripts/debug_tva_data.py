#!/usr/bin/env python3
"""Debug script to investigate TVA's data in Neo4j"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_tva_data():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        return
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # 1. Check TVA agency node
            logger.info("\nChecking TVA agency node:")
            result = session.run("""
                MATCH (a:Agency)
                WHERE a.abbreviation = 'TVA'
                RETURN a.name, a.abbreviation
            """)
            for record in result:
                logger.info(f"Found agency: {record['a.name']} ({record['a.abbreviation']})")
            
            # 2. Count TVA use cases
            logger.info("\nCounting TVA use cases:")
            result = session.run("""
                MATCH (a:Agency {abbreviation: 'TVA'})-[:HAS_USE_CASE]->(u:UseCase)
                RETURN count(u) as use_case_count
            """)
            logger.info(f"Total use cases: {result.single()['use_case_count']}")
            
            # 3. Sample TVA use cases
            logger.info("\nSample TVA use cases:")
            result = session.run("""
                MATCH (a:Agency {abbreviation: 'TVA'})-[:HAS_USE_CASE]->(u:UseCase)
                RETURN u.name
                LIMIT 5
            """)
            for record in result:
                logger.info(f"- {record['u.name']}")
            
            # 4. Check for evaluations
            logger.info("\nChecking for evaluations:")
            result = session.run("""
                MATCH (a:Agency {abbreviation: 'TVA'})-[:HAS_USE_CASE]->(u:UseCase)
                OPTIONAL MATCH (u)-[r:HAS_EVALUATION]->(e:CategoryEvaluation)
                RETURN 
                    count(DISTINCT u) as total_use_cases,
                    count(DISTINCT e) as total_evaluations
            """)
            record = result.single()
            logger.info(f"Use cases: {record['total_use_cases']}")
            logger.info(f"Evaluations: {record['total_evaluations']}")
            
            # 5. Sample evaluations with categories
            logger.info("\nSample evaluations with categories:")
            result = session.run("""
                MATCH (a:Agency {abbreviation: 'TVA'})-[:HAS_USE_CASE]->(u:UseCase)
                MATCH (u)-[r:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
                RETURN 
                    u.name as use_case,
                    c.name as category,
                    e.relationship_type as rel_type,
                    e.confidence as confidence
                LIMIT 5
            """)
            for record in result:
                logger.info(
                    f"Use Case: {record['use_case']}\n"
                    f"  Category: {record['category']}\n"
                    f"  Relationship: {record['rel_type']}\n"
                    f"  Confidence: {record['confidence']}"
                )
        
        driver.close()
        
    except Exception as e:
        logger.error(f"Error debugging TVA data: {str(e)}")

if __name__ == "__main__":
    debug_tva_data() 