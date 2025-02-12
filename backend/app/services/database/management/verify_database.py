"""
Dell-AITC Database Verification Script (v2.2)
Verifies that the Neo4j database is properly initialized and populated
by checking node counts, relationships, and data integrity.

Usage:
    python -m backend.app.services.database.management.verify_database

Environment Variables Required:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Configure logging
log_dir = Path("logs/database")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'database_verify_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
project_root = Path(__file__).parents[5]
load_dotenv(project_root / '.env')

# Neo4j configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

class DatabaseVerifier:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def check_node_counts(self) -> List[Dict[str, Any]]:
        """Verify the count of each node type."""
        queries = [
            ("Zone", "MATCH (z:Zone) RETURN count(z) as count"),
            ("AICategory", "MATCH (c:AICategory) RETURN count(c) as count"),
            ("Keyword", "MATCH (k:Keyword) RETURN count(k) as count"),
            ("UseCase", "MATCH (u:UseCase) RETURN count(u) as count"),
            ("Agency", "MATCH (a:Agency) RETURN count(a) as count"),
            ("Bureau", "MATCH (b:Bureau) RETURN count(b) as count")
        ]
        
        results = []
        with self.driver.session() as session:
            for label, query in queries:
                count = session.run(query).single()['count']
                results.append({
                    'label': label,
                    'count': count,
                    'status': 'OK' if count > 0 else 'EMPTY'
                })
        return results
        
    def check_relationships(self) -> List[Dict[str, Any]]:
        """Verify the relationships between nodes."""
        queries = [
            ("BELONGS_TO", "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count"),
            ("HAS_KEYWORD", "MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) as count"),
            ("IMPLEMENTED_BY", "MATCH ()-[r:IMPLEMENTED_BY]->() RETURN count(r) as count"),
            ("MANAGED_BY", "MATCH ()-[r:MANAGED_BY]->() RETURN count(r) as count"),
            ("HAS_BUREAU", "MATCH ()-[r:HAS_BUREAU]->() RETURN count(r) as count")
        ]
        
        results = []
        with self.driver.session() as session:
            for rel_type, query in queries:
                count = session.run(query).single()['count']
                results.append({
                    'relationship': rel_type,
                    'count': count,
                    'status': 'OK' if count > 0 else 'EMPTY'
                })
        return results
        
    def check_data_integrity(self) -> List[Dict[str, Any]]:
        """Verify data integrity rules."""
        checks = [
            {
                'name': 'Categories have zones',
                'query': """
                MATCH (c:AICategory)
                WHERE NOT (c)-[:BELONGS_TO]->(:Zone)
                RETURN count(c) as count
                """,
                'should_be_zero': True
            },
            {
                'name': 'Use cases have agencies',
                'query': """
                MATCH (u:UseCase)
                WHERE NOT (u)-[:IMPLEMENTED_BY]->(:Agency)
                RETURN count(u) as count
                """,
                'should_be_zero': True
            },
            {
                'name': 'Required properties on Use Cases',
                'query': """
                MATCH (u:UseCase)
                WHERE u.id IS NULL OR u.name IS NULL 
                    OR u.description IS NULL OR u.created_at IS NULL
                RETURN count(u) as count
                """,
                'should_be_zero': True
            },
            {
                'name': 'Required properties on Categories',
                'query': """
                MATCH (c:AICategory)
                WHERE c.id IS NULL OR c.name IS NULL 
                    OR c.category_definition IS NULL OR c.created_at IS NULL
                RETURN count(c) as count
                """,
                'should_be_zero': True
            }
        ]
        
        results = []
        with self.driver.session() as session:
            for check in checks:
                count = session.run(check['query']).single()['count']
                results.append({
                    'check': check['name'],
                    'violations': count,
                    'status': 'OK' if (count == 0) == check['should_be_zero'] else 'FAILED'
                })
        return results

def main():
    """Run all database verifications."""
    try:
        verifier = DatabaseVerifier(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Check node counts
        logger.info("\n=== Node Counts ===")
        for result in verifier.check_node_counts():
            logger.info(f"{result['label']}: {result['count']} [{result['status']}]")
            
        # Check relationships
        logger.info("\n=== Relationships ===")
        for result in verifier.check_relationships():
            logger.info(f"{result['relationship']}: {result['count']} [{result['status']}]")
            
        # Check data integrity
        logger.info("\n=== Data Integrity ===")
        for result in verifier.check_data_integrity():
            logger.info(f"{result['check']}: {result['violations']} violation(s) [{result['status']}]")
            
        verifier.close()
        logger.info("\n[SUCCESS] Database verification completed")
        
    except Exception as e:
        logger.error(f"[ERROR] Database verification failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 