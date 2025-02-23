"""
Script to update Neo4j schema with missing properties.

Usage:
    python update_schema.py
"""

import os
import logging
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_schema_update():
    """Run the schema update script."""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    uri = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")
    
    # Read schema update script
    script_path = Path(__file__).parent / 'update_schema.cypher'
    if not script_path.exists():
        raise FileNotFoundError(f"Schema update script not found at {script_path}")
        
    with open(script_path, 'r') as f:
        cypher_script = f.read()
    
    # Connect to Neo4j and run script
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            logger.info("Running schema update script...")
            
            # Split script into individual statements
            statements = [stmt.strip() for stmt in cypher_script.split(';') if stmt.strip()]
            
            # Execute each statement
            for stmt in statements:
                logger.info(f"Executing: {stmt[:100]}...")  # Log first 100 chars
                result = session.run(stmt)
                summary = result.consume()
                logger.info(f"Statement completed: {summary.counters}")
                
        logger.info("Schema update completed successfully")
        
    except Exception as e:
        logger.error(f"Error updating schema: {str(e)}")
        raise
        
    finally:
        driver.close()

if __name__ == '__main__':
    run_schema_update() 