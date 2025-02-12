"""
Dell-AITC Neo4j Database Initialization Script (v2.2)
Handles complete database initialization and validation according to v2.2 schema

Usage:
    python -m backend.app.services.database.management.initialize_db

Environment Variables Required:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = Path("logs/database")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'neo4j_init_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

# Cypher statements for initialization
INIT_CONSTRAINTS = """
// Node uniqueness constraints
CREATE CONSTRAINT unique_category_id IF NOT EXISTS FOR (c:AICategory) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT unique_zone_id IF NOT EXISTS FOR (z:Zone) REQUIRE z.id IS UNIQUE;
CREATE CONSTRAINT unique_keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;
CREATE CONSTRAINT unique_usecase_id IF NOT EXISTS FOR (u:UseCase) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT unique_agency_abbrev IF NOT EXISTS FOR (a:Agency) REQUIRE a.abbreviation IS UNIQUE;
CREATE CONSTRAINT unique_bureau_id IF NOT EXISTS FOR (b:Bureau) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT unique_classification_id IF NOT EXISTS FOR (c:AIClassification) REQUIRE c.id IS UNIQUE;

// Property existence constraints
CREATE CONSTRAINT require_match_type IF NOT EXISTS 
FOR ()-[r:USES_TECHNOLOGY]-() 
REQUIRE r.match_type IS NOT NULL;

CREATE CONSTRAINT require_review_status IF NOT EXISTS 
FOR (c:AIClassification)
REQUIRE c.review_status IS NOT NULL;

CREATE CONSTRAINT require_analysis_method IF NOT EXISTS 
FOR (c:AIClassification)
REQUIRE c.analysis_method IS NOT NULL;

CREATE CONSTRAINT require_nomatch_status IF NOT EXISTS 
FOR (n:NoMatchAnalysis)
REQUIRE n.status IS NOT NULL;
"""

INIT_INDEXES = """
// B-tree indexes
CREATE INDEX category_name IF NOT EXISTS FOR (c:AICategory) ON (c.name);
CREATE INDEX zone_name IF NOT EXISTS FOR (z:Zone) ON (z.name);
CREATE INDEX keyword_name IF NOT EXISTS FOR (k:Keyword) ON (k.name);
CREATE INDEX usecase_name IF NOT EXISTS FOR (u:UseCase) ON (u.name);
CREATE INDEX agency_name IF NOT EXISTS FOR (a:Agency) ON (a.name);
CREATE INDEX bureau_name IF NOT EXISTS FOR (b:Bureau) ON (b.name);
CREATE INDEX classification_status IF NOT EXISTS FOR (c:AIClassification) ON (c.status);
CREATE INDEX classification_type IF NOT EXISTS FOR (c:AIClassification) ON (c.type);
CREATE INDEX agency_abbreviation IF NOT EXISTS FOR (a:Agency) ON (a.abbreviation);
CREATE INDEX nomatch_status IF NOT EXISTS FOR (n:NoMatchAnalysis) ON (n.status);

// Full-text indexes
CREATE FULLTEXT INDEX usecase_text IF NOT EXISTS 
FOR (u:UseCase) ON EACH [u.name, u.description, u.purpose_benefits];
CREATE FULLTEXT INDEX category_definition IF NOT EXISTS 
FOR (c:AICategory) ON EACH [c.definition];

// Add new indexes:
CREATE INDEX classification_review_status IF NOT EXISTS 
FOR (c:AIClassification) ON (c.review_status);

CREATE INDEX classification_analysis_method IF NOT EXISTS 
FOR (c:AIClassification) ON (c.analysis_method);
"""

# Define all constraint queries
constraint_queries = [
    # Node uniqueness constraints
    "CREATE CONSTRAINT unique_category_id IF NOT EXISTS FOR (c:AICategory) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT unique_zone_id IF NOT EXISTS FOR (z:Zone) REQUIRE z.id IS UNIQUE",
    "CREATE CONSTRAINT unique_keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
    "CREATE CONSTRAINT unique_usecase_id IF NOT EXISTS FOR (u:UseCase) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT unique_agency_abbrev IF NOT EXISTS FOR (a:Agency) REQUIRE a.abbreviation IS UNIQUE",
    "CREATE CONSTRAINT unique_bureau_id IF NOT EXISTS FOR (b:Bureau) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT unique_classification_id IF NOT EXISTS FOR (c:AIClassification) REQUIRE c.id IS UNIQUE"
]

# Define all index queries
index_queries = [
    # B-tree indexes
    "CREATE INDEX category_name IF NOT EXISTS FOR (c:AICategory) ON (c.name)",
    "CREATE INDEX zone_name IF NOT EXISTS FOR (z:Zone) ON (z.name)",
    "CREATE INDEX keyword_name IF NOT EXISTS FOR (k:Keyword) ON (k.name)",
    "CREATE INDEX usecase_name IF NOT EXISTS FOR (u:UseCase) ON (u.name)",
    "CREATE INDEX agency_name IF NOT EXISTS FOR (a:Agency) ON (a.name)",
    "CREATE INDEX bureau_name IF NOT EXISTS FOR (b:Bureau) ON (b.name)",
    "CREATE INDEX classification_status IF NOT EXISTS FOR (c:AIClassification) ON (c.status)",
    "CREATE INDEX classification_type IF NOT EXISTS FOR (c:AIClassification) ON (c.type)",
    "CREATE INDEX agency_abbreviation IF NOT EXISTS FOR (a:Agency) ON (a.abbreviation)",
    "CREATE INDEX nomatch_status IF NOT EXISTS FOR (n:NoMatchAnalysis) ON (n.status)",
    "CREATE INDEX classification_review_status IF NOT EXISTS FOR (c:AIClassification) ON (c.review_status)",
    "CREATE INDEX classification_analysis_method IF NOT EXISTS FOR (c:AIClassification) ON (c.analysis_method)",
    
    # Full-text indexes
    "CREATE FULLTEXT INDEX usecase_text IF NOT EXISTS FOR (u:UseCase) ON EACH [u.name, u.description, u.purpose_benefits]",
    "CREATE FULLTEXT INDEX category_definition IF NOT EXISTS FOR (c:AICategory) ON EACH [c.definition]"
]

VALIDATION_QUERIES = [
    {
        'name': 'Constraint Validation',
        'query': """
        SHOW CONSTRAINTS
        YIELD name
        RETURN 
            count(*) > 0 AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_category_id') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_zone_id') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_keyword_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_usecase_id') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_agency_abbrev') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_bureau_id') AND
            any(x IN collect(name) WHERE x CONTAINS 'unique_classification_id') AS all_constraints_exist
        """
    },
    {
        'name': 'Index Validation',
        'query': """
        SHOW INDEXES
        YIELD name
        RETURN 
            count(*) > 0 AND
            any(x IN collect(name) WHERE x CONTAINS 'category_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'zone_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'keyword_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'usecase_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'agency_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'bureau_name') AND
            any(x IN collect(name) WHERE x CONTAINS 'classification_status') AND
            any(x IN collect(name) WHERE x CONTAINS 'classification_type') AND
            any(x IN collect(name) WHERE x CONTAINS 'agency_abbreviation') AND
            any(x IN collect(name) WHERE x CONTAINS 'nomatch_status') AND
            any(x IN collect(name) WHERE x CONTAINS 'usecase_text') AND
            any(x IN collect(name) WHERE x CONTAINS 'category_definition') AS all_indexes_exist
        """
    }
]

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, message="Executing query"):
        logger.info(message)
        with self.driver.session() as session:
            try:
                result = session.run(query)
                return result.consume().counters
            except Exception as e:
                logger.error(f"Error executing query: {str(e)}")
                raise

    def execute_multiple_queries(self, queries, prefix_message="Executing queries"):
        """Execute multiple queries in sequence, logging each one."""
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"{prefix_message} ({i}/{len(queries)})")
                with self.driver.session() as session:
                    result = session.run(query)
                    result.consume()
            except Exception as e:
                logger.error(f"Error executing query {i}: {str(e)}")
                raise

    def validate_schema(self):
        validation_results = []
        with self.driver.session() as session:
            for validation in VALIDATION_QUERIES:
                try:
                    result = session.run(validation['query'])
                    records = result.single()
                    validation_results.append({
                        'name': validation['name'],
                        'success': records[0] if records else False
                    })
                except Exception as e:
                    validation_results.append({
                        'name': validation['name'],
                        'success': False,
                        'error': str(e)
                    })
        return validation_results

def initialize_database():
    """Initialize the Neo4j database with schema and constraints"""
    logger.info("Starting database initialization")
    
    try:
        neo4j = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Initialize constraints
        logger.info("Creating constraints...")
        neo4j.execute_multiple_queries(constraint_queries, "Creating constraint")
        
        # Initialize indexes
        logger.info("Creating indexes...")
        neo4j.execute_multiple_queries(index_queries, "Creating index")
        
        # Validate schema
        logger.info("Validating schema...")
        validation_results = neo4j.validate_schema()
        
        # Log validation results
        for result in validation_results:
            status = "✓ Passed" if result['success'] else "✗ Failed"
            logger.info(f"{result['name']}: {status}")
            if not result['success'] and 'error' in result:
                logger.error(f"Validation Error: {result['error']}")
        
        neo4j.close()
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        initialize_database()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1) 