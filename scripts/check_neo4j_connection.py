from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_neo4j_connection():
    """Check Neo4j connection and query execution"""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials from environment variables
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        return False
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connection
        with driver.session() as session:
            # Test basic query
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            logger.info(f"Total nodes in database: {count}")
            
            # Check for CategoryEvaluation nodes
            result = session.run("""
                MATCH (e:CategoryEvaluation)
                RETURN count(e) as eval_count
            """)
            eval_count = result.single()["eval_count"]
            logger.info(f"CategoryEvaluation nodes: {eval_count}")
            
            # Check for HAS_EVALUATION relationships
            result = session.run("""
                MATCH ()-[r:HAS_EVALUATION]->()
                RETURN count(r) as rel_count
            """)
            rel_count = result.single()["rel_count"]
            logger.info(f"HAS_EVALUATION relationships: {rel_count}")
            
            # Check for EVALUATES relationships
            result = session.run("""
                MATCH ()-[r:EVALUATES]->()
                RETURN count(r) as rel_count
            """)
            rel_count = result.single()["rel_count"]
            logger.info(f"EVALUATES relationships: {rel_count}")
            
            # Check for specific use cases with high confidence matches
            result = session.run("""
                MATCH (u:UseCase)-[r:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
                WHERE e.confidence >= 0.8
                RETURN u.name as use_case, c.name as category, e.confidence as confidence
                LIMIT 5
            """)
            logger.info("\nSample high confidence matches:")
            for record in result:
                logger.info(f"  {record['use_case']} -> {record['category']} (confidence: {record['confidence']:.2f})")
                
        driver.close()
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        return False

if __name__ == "__main__":
    check_neo4j_connection() 