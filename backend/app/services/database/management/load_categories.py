"""
Dell-AITC AI Technology Categories Loader (v2.2)
Loads AI technology categories from CSV file into Neo4j database.

Usage:
    python -m backend.app.services.database.management.load_categories

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
log_file = log_dir / f'categories_load_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

class CategoryLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def load_categories(self, categories_file: Path) -> bool:
        """Load AI technology categories from CSV file."""
        try:
            df = pd.read_csv(categories_file)
            
            # Validate required columns
            required_columns = {
                'ai_category', 'definition', 'zone',
                'keywords', 'capabilities', 'maturity_level'
            }
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
            # Create categories and relationships
            category_query = """
            MATCH (z:Zone {name: $zone})
            MERGE (c:AICategory {name: $name})
            SET c.id = $id,
                c.category_definition = $definition,
                c.status = 'active',
                c.maturity_level = $maturity_level,
                c.version = '1.0.0',
                c.created_at = CASE 
                    WHEN c.created_at IS NULL THEN datetime() 
                    ELSE c.created_at 
                END,
                c.last_updated = datetime()
            MERGE (c)-[r:BELONGS_TO]->(z)
            SET r.created_at = datetime(),
                r.weight = 1.0
            """
            
            # Create keywords and relationships
            keyword_query = """
            MERGE (k:Keyword {name: $keyword})
            SET k.id = $id,
                k.type = $type,
                k.created_at = CASE 
                    WHEN k.created_at IS NULL THEN datetime() 
                    ELSE k.created_at 
                END,
                k.last_updated = datetime()
            WITH k
            MATCH (c:AICategory {name: $category})
            MERGE (c)-[r:HAS_KEYWORD]->(k)
            SET r.created_at = datetime()
            """
            
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    # Create category
                    session.run(category_query, {
                        'name': row['ai_category'],
                        'definition': row['definition'],
                        'zone': row['zone'],
                        'maturity_level': row['maturity_level'],
                        'id': str(uuid.uuid4())
                    })
                    
                    # Process keywords
                    if pd.notna(row['keywords']):
                        keywords = [k.strip() for k in row['keywords'].split(';')]
                        for keyword in keywords:
                            session.run(keyword_query, {
                                'keyword': keyword, 
                                'category': row['ai_category'],
                                'type': 'technical_keywords',
                                'id': str(uuid.uuid4())
                            })
                    
                    # Process capabilities
                    if pd.notna(row['capabilities']):
                        capabilities = [c.strip() for c in row['capabilities'].split(';')]
                        for capability in capabilities:
                            session.run(keyword_query, {
                                'keyword': capability,
                                'category': row['ai_category'],
                                'type': 'capabilities',
                                'id': str(uuid.uuid4())
                            })
            
            logger.info(f"Successfully loaded categories from {categories_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load categories: {str(e)}")
            return False

def main():
    """Main entry point for category loading."""
    try:
        # Find the categories file
        data_dir = Path("data/input")
        categories_file = next(data_dir.glob("AI-Technology-Categories-v*.csv"), None)
        
        if not categories_file:
            raise FileNotFoundError("No AI Technology Categories file found")
            
        logger.info(f"Found categories file: {categories_file}")
        
        # Load the categories
        loader = CategoryLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        success = loader.load_categories(categories_file)
        loader.close()
        
        if success:
            logger.info("[SUCCESS] Category loading completed successfully")
        else:
            logger.error("[FAILED] Category loading failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 