"""
Verify database initialization and data loading.

Usage:
    python scripts/verify_initialization.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add backend to Python path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / 'backend'
sys.path.append(str(backend_dir))

from app.services.database.neo4j_service import Neo4jService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def verify_schema(service: Neo4jService) -> bool:
    """Verify schema constraints and indexes."""
    try:
        # Check constraints
        result = await service.run_query("""
        SHOW CONSTRAINTS
        YIELD name, type
        RETURN collect(name) as names
        """)
        constraints = result[0]['names']
        required_constraints = [
            'unique_category_id',
            'unique_zone_id',
            'unique_keyword_name',
            'unique_usecase_id',
            'unique_agency_abbrev',
            'unique_bureau_id',
            'unique_classification_id'
        ]
        
        missing_constraints = [c for c in required_constraints if not any(c in x for x in constraints)]
        if missing_constraints:
            logger.error(f"Missing constraints: {missing_constraints}")
            return False
            
        # Check indexes
        result = await service.run_query("""
        SHOW INDEXES
        YIELD name
        RETURN collect(name) as names
        """)
        indexes = result[0]['names']
        required_indexes = [
            'category_name',
            'zone_name',
            'keyword_name',
            'usecase_name',
            'agency_name',
            'bureau_name',
            'classification_status',
            'classification_type',
            'nomatch_status',
            'usecase_text',
            'category_definition'
        ]
        
        # Note: agency_abbreviation index is covered by unique_agency_abbrev constraint
        missing_indexes = []
        for idx in required_indexes:
            if not any(idx in x for x in indexes):
                missing_indexes.append(idx)
                
        if missing_indexes:
            logger.error(f"Missing indexes: {missing_indexes}")
            return False
            
        # Log successful verification
        logger.info("Schema verification passed:")
        logger.info(f"- All {len(required_constraints)} required constraints present")
        logger.info(f"- All {len(required_indexes)} required indexes present")
        logger.info("- Agency abbreviation indexing verified (via unique constraint)")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema verification failed: {str(e)}")
        return False

async def verify_data(service: Neo4jService) -> bool:
    """Verify data loading."""
    try:
        # Check node counts
        counts = {}
        for label in ['Zone', 'AICategory', 'UseCase', 'Agency', 'Bureau', 'Keyword']:
            result = await service.run_query(f"MATCH (n:{label}) RETURN count(n) as count")
            counts[label] = result[0]['count']
            
        # Verify minimum counts
        requirements = {
            'Zone': 3,  # At least 3 zones
            'AICategory': 14,  # Exactly 14 categories
            'UseCase': 100,  # At least 100 use cases
            'Agency': 5,  # At least 5 agencies
            'Keyword': 10  # At least 10 keywords
        }
        
        for label, min_count in requirements.items():
            if counts[label] < min_count:
                logger.error(f"Insufficient {label} count. Found: {counts[label]}, Required: {min_count}")
                return False
                
        # Verify relationships
        relationships = [
            'BELONGS_TO',  # Categories -> Zones
            'HAS_KEYWORD',  # Categories -> Keywords
            'IMPLEMENTED_BY',  # UseCases -> Agencies
            'MANAGED_BY',  # UseCases -> Bureaus
            'HAS_BUREAU'  # Agencies -> Bureaus
        ]
        
        for rel in relationships:
            result = await service.run_query(f"MATCH ()-[r:{rel}]->() RETURN count(r) as count")
            count = result[0]['count']
            if count == 0:
                logger.error(f"No {rel} relationships found")
                return False
                
        # Verify data integrity
        # Check for orphaned nodes
        orphans = await service.run_query("""
        MATCH (n) 
        WHERE NOT (n)--() 
        RETURN labels(n) as label, count(*) as count
        """)
        
        if any(r['count'] > 0 for r in orphans):
            logger.error("Found orphaned nodes:")
            for r in orphans:
                if r['count'] > 0:
                    logger.error(f"  {r['label']}: {r['count']}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Data verification failed: {str(e)}")
        return False

async def verify_keyword_relevance(service: Neo4jService) -> bool:
    """Verify keyword relevance properties."""
    try:
        # Check Keyword nodes have relevance_score
        result = await service.run_query("""
        MATCH (k:Keyword)
        WHERE k.relevance_score IS NULL
        RETURN count(k) as count
        """)
        if result[0]['count'] > 0:
            logger.error(f"Found {result[0]['count']} keywords without relevance_score")
            return False
            
        # Check HAS_KEYWORD relationships have relevance
        result = await service.run_query("""
        MATCH ()-[r:HAS_KEYWORD]->()
        WHERE r.relevance IS NULL
        RETURN count(r) as count
        """)
        if result[0]['count'] > 0:
            logger.error(f"Found {result[0]['count']} HAS_KEYWORD relationships without relevance")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Keyword relevance verification failed: {str(e)}")
        return False

async def main():
    """Run all verifications."""
    try:
        service = Neo4jService()
        
        logger.info("Verifying schema...")
        schema_ok = await verify_schema(service)
        
        logger.info("Verifying data...")
        data_ok = await verify_data(service)
        
        logger.info("Verifying keyword relevance...")
        relevance_ok = await verify_keyword_relevance(service)
        
        await service.cleanup()
        
        if schema_ok and data_ok and relevance_ok:
            logger.info("\n[SUCCESS] Database initialization verified successfully!")
            return True
        else:
            logger.error("\n[FAILED] Database initialization verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 