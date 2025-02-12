#!/usr/bin/env python
"""
Neo4j Schema Migration Script for v2.2
Applies schema changes for enhanced technology classification.

This script:
1. Creates new constraints and indexes
2. Updates existing relationships with new properties
3. Validates the migration
4. Provides rollback capability
"""

import logging
from neo4j import GraphDatabase
from pathlib import Path
import json
import argparse
from typing import List, Dict
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaMigration:
    def __init__(self):
        # Get settings from environment variables
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD', 'password')
            )
        )
        
    def __del__(self):
        self.driver.close()
        
    def backup_relationships(self):
        """Create backup of existing relationships"""
        with self.driver.session() as session:
            logger.info("Creating relationship backups...")
            query = """
            MATCH (u:UseCase)-[r:USES_TECHNOLOGY]->(c:AICategory)
            WITH u, c, properties(r) as props
            MERGE (u)-[new:USES_TECHNOLOGY_V2_2_BACKUP]->(c)
            SET new = props
            """
            session.run(query)
            logger.info("Relationship backup completed")
            
    def create_constraints(self):
        """Create new constraints for v2.2 schema"""
        with self.driver.session() as session:
            logger.info("Creating new constraints...")
            
            # Drop existing constraints if they exist
            try:
                session.run("DROP CONSTRAINT unique_use_case_primary_match IF EXISTS")
                session.run("DROP CONSTRAINT aiclassification_id IF EXISTS")
                session.run("DROP CONSTRAINT nomatchanalysis_id IF EXISTS")
            except Exception as e:
                logger.warning(f"Failed to drop constraints: {str(e)}")
            
            # Create new constraints
            session.run("""
            CREATE CONSTRAINT aiclassification_id IF NOT EXISTS 
            FOR (n:AIClassification) REQUIRE n.id IS UNIQUE
            """)
            
            session.run("""
            CREATE CONSTRAINT nomatchanalysis_id IF NOT EXISTS 
            FOR (n:NoMatchAnalysis) REQUIRE n.id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
            CREATE INDEX aiclassification_match_type IF NOT EXISTS 
            FOR (n:AIClassification) ON (n.match_type)
            """)
            
            session.run("""
            CREATE INDEX aiclassification_confidence IF NOT EXISTS 
            FOR (n:AIClassification) ON (n.confidence)
            """)
            
            session.run("""
            CREATE INDEX aiclassification_review_status IF NOT EXISTS 
            FOR (n:AIClassification) ON (n.review_status)
            """)
            
            session.run("""
            CREATE INDEX nomatchanalysis_status IF NOT EXISTS 
            FOR (n:NoMatchAnalysis) ON (n.status)
            """)
            
            logger.info("Constraints and indexes created successfully")
            
    def update_relationships(self):
        """Update existing relationships with new properties and create new nodes"""
        with self.driver.session() as session:
            logger.info("Updating existing relationships...")
            
            # Convert USES_TECHNOLOGY relationships to AIClassification nodes
            query = """
            MATCH (u:UseCase)-[r:USES_TECHNOLOGY]->(c:AICategory)
            
            // Create AIClassification node
            CREATE (cl:AIClassification {
                id: apoc.create.uuid(),
                match_type: CASE 
                    WHEN r.confidence >= 0.45 THEN 'PRIMARY'
                    WHEN r.confidence >= 0.35 THEN 'SUPPORTING'
                    ELSE 'RELATED'
                END,
                confidence: r.confidence,
                analysis_method: CASE
                    WHEN r.match_method = 'keyword' THEN 'KEYWORD'
                    WHEN r.match_method = 'semantic' THEN 'SEMANTIC'
                    WHEN r.match_method = 'ensemble_high_agreement' THEN 'ENSEMBLE'
                    ELSE 'KEYWORD'
                END,
                analysis_version: 'v2.2',
                keyword_score: r.keyword_score,
                semantic_score: r.semantic_score,
                llm_score: r.llm_score,
                field_match_scores: CASE 
                    WHEN r.matched_keywords IS NOT NULL 
                    THEN {keyword_matches: r.matched_keywords}
                    ELSE {}
                END,
                term_match_details: r.match_details,
                matched_keywords: r.matched_keywords,
                llm_verification: false,
                llm_confidence: 0.0,
                llm_reasoning: '',
                llm_suggestions: [],
                improvement_notes: [],
                false_positive: false,
                manual_override: false,
                review_status: 'PENDING',
                classified_at: datetime(),
                classified_by: 'system',
                last_updated: datetime()
            })
            
            // Create new relationships
            CREATE (c)-[:CLASSIFIES]->(cl)
            CREATE (u)-[:CLASSIFIED_AS]->(cl)
            
            // Delete old relationship
            DELETE r
            """
            session.run(query)
            
            # Create NoMatchAnalysis nodes for unmatched use cases
            query = """
            MATCH (u:UseCase)
            WHERE NOT (u)-[:CLASSIFIED_AS]->(:AIClassification)
            CREATE (na:NoMatchAnalysis {
                id: apoc.create.uuid(),
                reason: 'No matching category found',
                confidence: 0.0,
                llm_analysis: {},
                suggested_keywords: [],
                improvement_suggestions: {},
                created_at: datetime(),
                analyzed_by: 'system',
                status: 'NEW',
                review_notes: ''
            })
            CREATE (u)-[:HAS_ANALYSIS]->(na)
            """
            session.run(query)
            
            logger.info("Relationships and nodes updated successfully")
            
    def validate_migration(self) -> bool:
        """Validate the migration was successful"""
        with self.driver.session() as session:
            logger.info("Validating migration...")
            
            # Check for AIClassification nodes
            query = """
            MATCH (cl:AIClassification)
            WHERE NOT exists(cl.id) 
               OR NOT exists(cl.match_type)
               OR NOT exists(cl.confidence)
               OR NOT exists(cl.analysis_method)
            RETURN count(cl) as missing_props
            """
            result = session.run(query)
            missing = result.single()['missing_props']
            
            if missing > 0:
                logger.error(f"Found {missing} AIClassification nodes with missing required properties")
                return False
                
            # Check for NoMatchAnalysis nodes
            query = """
            MATCH (na:NoMatchAnalysis)
            WHERE NOT exists(na.id)
               OR NOT exists(na.reason)
               OR NOT exists(na.status)
            RETURN count(na) as missing_props
            """
            result = session.run(query)
            missing = result.single()['missing_props']
            
            if missing > 0:
                logger.error(f"Found {missing} NoMatchAnalysis nodes with missing required properties")
                return False
                
            # Check for orphaned nodes
            query = """
            MATCH (cl:AIClassification)
            WHERE NOT (cl)<-[:CLASSIFIES]-(:AICategory)
               OR NOT (cl)<-[:CLASSIFIED_AS]-(:UseCase)
            RETURN count(cl) as orphaned
            """
            result = session.run(query)
            orphaned = result.single()['orphaned']
            
            if orphaned > 0:
                logger.error(f"Found {orphaned} orphaned AIClassification nodes")
                return False
                
            logger.info("Migration validation successful")
            return True
            
    def rollback(self):
        """Rollback changes if needed"""
        with self.driver.session() as session:
            logger.info("Rolling back changes...")
            
            # Delete new nodes and relationships
            query = """
            MATCH (cl:AIClassification)
            OPTIONAL MATCH (cl)<-[r1:CLASSIFIES]-()
            OPTIONAL MATCH (cl)<-[r2:CLASSIFIED_AS]-()
            DELETE r1, r2, cl
            """
            session.run(query)
            
            query = """
            MATCH (na:NoMatchAnalysis)
            OPTIONAL MATCH (na)<-[r:HAS_ANALYSIS]-()
            DELETE r, na
            """
            session.run(query)
            
            # Restore from backup
            query = """
            MATCH (u:UseCase)-[backup:USES_TECHNOLOGY_V2_2_BACKUP]->(c:AICategory)
            WITH u, c, properties(backup) as props
            CREATE (u)-[r:USES_TECHNOLOGY]->(c)
            SET r = props
            DELETE backup
            RETURN count(r) as restored
            """
            result = session.run(query)
            restored = result.single()['restored']
            logger.info(f"Restored {restored} relationships")
            
            # Drop new constraints and indexes
            try:
                session.run("DROP CONSTRAINT aiclassification_id IF EXISTS")
                session.run("DROP CONSTRAINT nomatchanalysis_id IF EXISTS")
                session.run("DROP INDEX aiclassification_match_type IF EXISTS")
                session.run("DROP INDEX aiclassification_confidence IF EXISTS")
                session.run("DROP INDEX aiclassification_review_status IF EXISTS")
                session.run("DROP INDEX nomatchanalysis_status IF EXISTS")
            except Exception as e:
                logger.warning(f"Failed to drop constraints/indexes: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Migrate Neo4j schema to v2.2')
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    args = parser.parse_args()
    
    migration = SchemaMigration()
    
    try:
        if args.rollback:
            migration.rollback()
        else:
            # Perform migration
            migration.backup_relationships()
            migration.create_constraints()
            migration.update_relationships()
            
            # Validate
            if not migration.validate_migration():
                logger.error("Migration validation failed")
                migration.rollback()
                return 1
                
            logger.info("Migration completed successfully")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        if not args.rollback:
            logger.info("Attempting rollback...")
            migration.rollback()
        return 1
        
    return 0

if __name__ == "__main__":
    main() 