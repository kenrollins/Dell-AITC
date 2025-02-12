"""
Dell-AITC AI Technology Zones Loader (v2.2)
Loads AI technology zones from CSV file into Neo4j database.

Usage:
    python -m backend.app.services.database.management.load_zones

Environment Variables Required:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import os
import uuid
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
log_dir = Path("logs/database")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'zones_load_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from project root .env
project_root = Path(__file__).parents[5]  # Navigate up to project root
load_dotenv(project_root / '.env')

# Neo4j configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not found in environment variables")

class ZoneLoader:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the zone loader with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Clean up the database connection."""
        self.driver.close()
        
    def load_zones(self, zones_file: Path) -> bool:
        """Load AI technology zones from CSV file.
        
        Args:
            zones_file: Path to the zones CSV file
            
        Returns:
            bool: True if loading was successful, False otherwise
            
        The zones file should contain the following columns:
        - ai_zone: Name of the technology zone
        - zone_definition: Description of the zone
        """
        try:
            df = pd.read_csv(zones_file)
            
            # Validate required columns
            required_columns = {'ai_zone', 'zone_definition'}
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
            # Create zones
            query = """
            UNWIND $zones as zone
            MERGE (z:Zone {name: zone.name})
            SET z.id = zone.id,
                z.description = zone.description,
                z.created_at = CASE 
                    WHEN z.created_at IS NULL THEN datetime() 
                    ELSE z.created_at 
                END,
                z.last_updated = datetime()
            """
            
            # Prepare zone data
            zones = [
                {
                    'name': row['ai_zone'],
                    'description': row['zone_definition'],
                    'id': str(uuid.uuid4())
                }
                for _, row in df.iterrows()
            ]
            
            # Execute the query
            with self.driver.session() as session:
                result = session.run(query, {'zones': zones})
                summary = result.consume()
                
                # Log the results
                logger.info(f"Created/Updated {summary.counters.nodes_created} new zones")
                logger.info(f"Modified {summary.counters.properties_set} properties")
                
            logger.info(f"Successfully loaded {len(zones)} zones from {zones_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load zones: {str(e)}")
            return False
            
    def verify_zones(self) -> bool:
        """Verify that zones were loaded correctly."""
        try:
            with self.driver.session() as session:
                # Check zone count
                result = session.run("MATCH (z:Zone) RETURN count(z) as count")
                zone_count = result.single()['count']
                logger.info(f"Found {zone_count} zones in database")
                
                # Check zone properties
                result = session.run("""
                MATCH (z:Zone)
                WHERE z.id IS NULL OR z.description IS NULL 
                    OR z.created_at IS NULL OR z.last_updated IS NULL
                RETURN count(z) as invalid_count
                """)
                invalid_count = result.single()['invalid_count']
                
                if invalid_count > 0:
                    logger.error(f"Found {invalid_count} zones with missing properties")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Zone verification failed: {str(e)}")
            return False

def main():
    """Main entry point for zone loading."""
    try:
        # Find the zones file
        data_dir = Path("data/input")
        zones_file = next(data_dir.glob("AI-Technology-zones-v*.csv"), None)
        
        if not zones_file:
            raise FileNotFoundError("No AI Technology Zones file found")
            
        logger.info(f"Found zones file: {zones_file}")
        
        # Load the zones
        loader = ZoneLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Load zones
        success = loader.load_zones(zones_file)
        if not success:
            logger.error("[FAILED] Zone loading failed")
            loader.close()
            exit(1)
            
        # Verify zones
        if not loader.verify_zones():
            logger.error("[FAILED] Zone verification failed")
            loader.close()
            exit(1)
            
        loader.close()
        logger.info("[SUCCESS] Zone loading and verification completed successfully")
        
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 