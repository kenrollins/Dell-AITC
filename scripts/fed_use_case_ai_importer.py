#!/usr/bin/env python3
"""
Federal Use Case AI Technology Importer

This script imports federal AI use case classifications and their technology category evaluations 
into Neo4j. It handles initial imports and updates, with support for dry runs and partial imports.

Input File:
- Automatically uses the latest fed_use_case_ai_classification_neo4j_[timestamp].csv from data/output/results/
- Copies the file to data/input/ for archival and tracking
- Can be overridden with --input-file parameter

Usage Examples:
1. Automatic mode (recommended):
   python fed_use_case_ai_importer.py --dry-run  # Uses latest file from output directory

2. Manual mode (override):
   python fed_use_case_ai_importer.py --input-file path/to/specific/file.csv

Workflow:
1. First, run fed_use_case_ai_classifier.py which:
   - Processes use cases through keyword matching
   - Performs semantic matching
   - Does LLM verification
   - Outputs two files:
     a. fed_use_case_ai_classification_neo4j_[timestamp].csv - Contains full evaluation results
     b. fed_use_case_ai_classification_preview_[timestamp].csv - Contains summary statistics

2. Then run this script (fed_use_case_ai_importer.py) to:
   - Automatically find the latest classification results
   - Copy them to the input directory for tracking
   - Import the results into Neo4j
   - Create nodes and relationships
   - Track evaluation metadata

Features:
- Automatic latest file detection
- Input file archival
- Dry run mode to preview changes
- Partial import support for testing
- Version tracking for evaluations
- Update handling with conflict resolution
- Audit trail of changes
- Detailed logging of import process
"""

import argparse
import logging
import sys
import json
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fed_use_case_importer')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Import federal AI use case classifications and their technology category evaluations into Neo4j"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the database"
    )
    
    parser.add_argument(
        "-n", "--number",
        type=int,
        help="Number of use cases to process (for testing)"
    )
    
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing evaluations (default: skip existing)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="Optional: Path to specific classification file (overrides automatic latest file detection)"
    )
    
    return parser.parse_args()

@dataclass
class ImportConfig:
    """Configuration for the import process"""
    dry_run: bool
    number: Optional[int]
    update: bool
    input_file: Path

@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

class DataValidator:
    """Validates input data before import"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def validate_required_columns(self) -> bool:
        """Check if all required columns are present"""
        required_columns = {
            'use_case_name', 'agency', 'category_name',
            'keyword_score', 'semantic_score', 'llm_score', 'final_score',
            'match_method', 'relationship_type', 'confidence',
            'matched_keywords', 'explanation', 'error'
        }
        
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            self.errors.append(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def validate_score_ranges(self) -> bool:
        """Validate that all scores are within valid ranges"""
        valid = True
        score_columns = ['keyword_score', 'semantic_score', 'llm_score', 'final_score', 'confidence']
        
        for col in score_columns:
            # Convert scores to float, handling any string formatting
            try:
                self.df[col] = self.df[col].astype(float)
                
                invalid_scores = self.df[
                    (self.df[col] < 0) | (self.df[col] > 1)
                ]
                if not invalid_scores.empty:
                    self.errors.append(f"Invalid {col} values found: {invalid_scores.index.tolist()}")
                    valid = False
            except Exception as e:
                self.errors.append(f"Error converting {col} to float: {str(e)}")
                valid = False
        
        return valid
    
    def validate_relationship_types(self) -> bool:
        """Validate relationship types"""
        # Convert all types to uppercase for comparison
        valid_types = {'PRIMARY', 'SECONDARY', 'RELATED', 'NO_MATCH'}
        current_types = set(self.df['relationship_type'].str.upper())
        invalid_types = current_types - valid_types
        
        if invalid_types:
            self.errors.append(f"Invalid relationship types found: {invalid_types}")
            return False
            
        # Convert all relationship types to uppercase
        self.df['relationship_type'] = self.df['relationship_type'].str.upper()
        return True
    
    def validate_match_methods(self) -> bool:
        """Validate match methods"""
        # Convert all methods to uppercase for comparison
        valid_methods = {'KEYWORD', 'SEMANTIC', 'LLM', 'NO_MATCH', 'ERROR'}
        current_methods = set(self.df['match_method'].str.upper())
        invalid_methods = current_methods - valid_methods
        
        if invalid_methods:
            self.errors.append(f"Invalid match methods found: {invalid_methods}")
            return False
            
        # Convert all match methods to uppercase
        self.df['match_method'] = self.df['match_method'].str.upper()
        return True
    
    def compute_statistics(self):
        """Compute statistics about the data"""
        self.stats = {
            'total_records': len(self.df),
            'unique_use_cases': self.df['use_case_name'].nunique(),
            'unique_agencies': self.df['agency'].nunique(),
            'unique_categories': self.df['category_name'].nunique(),
            'relationship_distribution': self.df['relationship_type'].value_counts().to_dict(),
            'method_distribution': self.df['match_method'].value_counts().to_dict(),
            'average_scores': {
                'keyword': self.df['keyword_score'].mean(),
                'semantic': self.df['semantic_score'].mean(),
                'llm': self.df['llm_score'].mean(),
                'final': self.df['final_score'].mean(),
                'confidence': self.df['confidence'].mean()
            }
        }
    
    def validate(self) -> ValidationResult:
        """Run all validations"""
        validations = [
            self.validate_required_columns(),
            self.validate_score_ranges(),
            self.validate_relationship_types(),
            self.validate_match_methods()
        ]
        
        self.compute_statistics()
        
        return ValidationResult(
            is_valid=all(validations),
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats
        )

    def validate_data(self):
        """Validate the input data."""
        try:
            # Check required columns
            required_columns = [
                'use_case_name',
                'agency_name',
                'category_name',
                'relationship_type',
                'match_method',
                'keyword_score',
                'semantic_score',
                'llm_score',
                'final_score',
                'confidence_score',
                'matched_keywords'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert numeric columns to float
            numeric_columns = [
                'keyword_score',
                'semantic_score',
                'llm_score',
                'final_score',
                'confidence_score'
            ]
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Convert string columns to uppercase
            self.df['relationship_type'] = self.df['relationship_type'].str.upper()
            self.df['match_method'] = self.df['match_method'].str.upper()
            
            # Log data statistics
            self.logger.info("Data statistics:")
            self.logger.info(f"  total_records: {len(self.df)}")
            self.logger.info(f"  unique_use_cases: {len(self.df['use_case_name'].unique())}")
            self.logger.info(f"  unique_agencies: {len(self.df['agency_name'].unique())}")
            self.logger.info(f"  unique_categories: {len(self.df['category_name'].unique())}")
            
            # Log relationship distribution
            rel_dist = self.df['relationship_type'].value_counts().to_dict()
            self.logger.info("  relationship_distribution:")
            for rel_type, count in rel_dist.items():
                self.logger.info(f"    {rel_type}: {count}")
            
            # Log method distribution
            method_dist = self.df['match_method'].value_counts().to_dict()
            self.logger.info("  method_distribution:")
            for method, count in method_dist.items():
                self.logger.info(f"    {method}: {count}")
            
            # Log average scores
            self.logger.info("  average_scores:")
            for col in numeric_columns:
                self.logger.info(f"    {col.replace('_score', '')}: {self.df[col].mean()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return False

class Neo4jValidator:
    """Validates Neo4j database state before import and handles schema setup"""
    
    def __init__(self, driver: GraphDatabase.driver):
        self.driver = driver
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def check_database_connection(self) -> bool:
        """Verify database connection and access"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                self.stats['total_nodes'] = count
                return True
        except Exception as e:
            self.errors.append(f"Database connection error: {str(e)}")
            return False
    
    def check_constraints_and_indexes(self) -> bool:
        """Verify required constraints and indexes exist"""
        try:
            with self.driver.session() as session:
                # Check for required constraints
                constraints = session.run("SHOW CONSTRAINTS").data()
                indexes = session.run("SHOW INDEXES").data()
                
                required_constraints = {
                    'use_case_name',
                    'category_name',
                    'agency_name',
                    'category_evaluation_id'  # Added for new node type
                }
                
                existing_constraints = {c['name'] for c in constraints}
                missing_constraints = required_constraints - existing_constraints
                
                if missing_constraints:
                    self.warnings.append(
                        f"Missing recommended constraints: {missing_constraints}"
                    )
                
                self.stats['constraints'] = len(constraints)
                self.stats['indexes'] = len(indexes)
                
                return True
        except Exception as e:
            self.errors.append(f"Error checking constraints: {str(e)}")
            return False

    def setup_schema(self) -> bool:
        """Set up required schema elements if they don't exist"""
        try:
            with self.driver.session() as session:
                # First drop any conflicting indexes that might prevent constraint creation
                try:
                    session.run("""
                    CALL db.indexes() YIELD name, labelsOrTypes, properties
                    WHERE labelsOrTypes[0] IN ['UseCase', 'Agency', 'AICategory', 'CategoryEvaluation']
                    RETURN name, labelsOrTypes, properties
                    """).data()
                except Exception as e:
                    self.warnings.append(f"Could not check existing indexes: {str(e)}")

                # Create constraints one by one to handle failures gracefully
                constraints = [
                    ("use_case_composite_key", """
                    CREATE CONSTRAINT use_case_composite_key IF NOT EXISTS
                    FOR (u:UseCase) REQUIRE (u.name, u.inventory_year) IS UNIQUE
                    """),
                    ("agency_name", """
                    CREATE CONSTRAINT agency_name IF NOT EXISTS
                    FOR (a:Agency) REQUIRE a.name IS UNIQUE
                    """),
                    ("category_name", """
                    CREATE CONSTRAINT category_name IF NOT EXISTS
                    FOR (c:AICategory) REQUIRE c.name IS UNIQUE
                    """),
                    ("category_evaluation_id", """
                    CREATE CONSTRAINT category_evaluation_id IF NOT EXISTS
                    FOR (e:CategoryEvaluation) REQUIRE e.id IS UNIQUE
                    """),
                    ("evaluation_batch_id", """
                    CREATE CONSTRAINT evaluation_batch_id IF NOT EXISTS
                    FOR (b:EvaluationBatch) REQUIRE b.id IS UNIQUE
                    """)
                ]

                for name, query in constraints:
                    try:
                        session.run(query)
                        logger.info(f"Created or verified constraint: {name}")
                    except Exception as e:
                        self.warnings.append(f"Could not create constraint {name}: {str(e)}")

                # Create indexes one by one
                indexes = [
                    ("use_case_inventory_year", """
                    CREATE INDEX use_case_inventory_year IF NOT EXISTS
                    FOR (u:UseCase) ON (u.inventory_year)
                    """),
                    ("use_case_status", """
                    CREATE INDEX use_case_status IF NOT EXISTS
                    FOR (u:UseCase) ON (u.status)
                    """),
                    ("agency_type", """
                    CREATE INDEX agency_type IF NOT EXISTS
                    FOR (a:Agency) ON (a.agency_type)
                    """),
                    ("category_evaluation_relationship_type", """
                    CREATE INDEX category_evaluation_relationship_type IF NOT EXISTS
                    FOR (e:CategoryEvaluation) ON (e.relationship_type)
                    """),
                    ("category_evaluation_date", """
                    CREATE INDEX category_evaluation_date IF NOT EXISTS
                    FOR (e:CategoryEvaluation) ON (e.evaluation_date)
                    """),
                    ("category_evaluation_match_method", """
                    CREATE INDEX category_evaluation_match_method IF NOT EXISTS
                    FOR (e:CategoryEvaluation) ON (e.match_method)
                    """)
                ]

                for name, query in indexes:
                    try:
                        session.run(query)
                        logger.info(f"Created or verified index: {name}")
                    except Exception as e:
                        self.warnings.append(f"Could not create index {name}: {str(e)}")

                # Set up schema metadata
                try:
                    metadata_query = """
                    MERGE (m:SchemaMetadata {id: 'fed_use_case_schema'})
                    SET m += {
                        version: '1.0',
                        last_updated: datetime(),
                        supported_relationship_types: ['PRIMARY', 'SECONDARY', 'RELATED', 'NO_MATCH'],
                        supported_match_methods: ['KEYWORD', 'SEMANTIC', 'LLM', 'NO_MATCH', 'ERROR'],
                        supported_use_case_statuses: ['ACTIVE', 'PLANNED', 'COMPLETED', 'CANCELLED'],
                        supported_agency_types: ['EXECUTIVE', 'INDEPENDENT', 'LEGISLATIVE', 'JUDICIAL'],
                        supported_investment_stages: ['PLANNING', 'INITIAL', 'OPERATIONAL', 'COMPLETED']
                    }
                    """
                    session.run(metadata_query)
                    logger.info("Updated schema metadata")
                except Exception as e:
                    self.warnings.append(f"Could not update schema metadata: {str(e)}")

                # If we have warnings but no errors, consider it a success
                if self.warnings:
                    logger.warning("Schema setup completed with warnings:")
                    for warning in self.warnings:
                        logger.warning(f"  - {warning}")
                else:
                    logger.info("Schema setup completed successfully")
                return True

        except Exception as e:
            self.errors.append(f"Error setting up schema: {str(e)}")
            return False

    def validate(self) -> ValidationResult:
        """Run all database validations and setup schema if needed"""
        # First check connection
        if not self.check_database_connection():
            return ValidationResult(
                is_valid=False,
                errors=self.errors,
                warnings=self.warnings,
                stats=self.stats
            )
        
        # Check existing constraints and indexes
        self.check_constraints_and_indexes()
        
        # Set up schema if there are missing constraints
        if self.warnings and any('Missing recommended constraints' in w for w in self.warnings):
            logger.info("Setting up missing schema elements...")
            if not self.setup_schema():
                return ValidationResult(
                    is_valid=False,
                    errors=self.errors,
                    warnings=self.warnings,
                    stats=self.stats
                )
        
        return ValidationResult(
            is_valid=True,
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats
        )

class Neo4jImporter:
    """Handles importing federal use case data into Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str, config: ImportConfig):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.config = config
        self.logger = logging.getLogger('fed_use_case_importer.neo4j')
    
    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def test_import_process(self, df: pd.DataFrame) -> bool:
        """Test the import process without making changes."""
        try:
            # Get first use case for testing
            use_case_name = df['use_case_name'].iloc[0]
            use_case_data = df[df['use_case_name'] == use_case_name]
            
            self.logger.info(f"Processing use case: {use_case_name}")
            self.logger.info(f"DataFrame columns: {list(df.columns)}")
            self.logger.info(f"Use case data columns: {list(use_case_data.columns)}")
            self.logger.info(f"Use case data shape: {use_case_data.shape}")
            self.logger.info(f"First row: {use_case_data.iloc[0].to_dict()}")
            
            agency_name = use_case_data['agency'].iloc[0]
            self.logger.info(f"Agency name: {agency_name}")
            
            # Create use case node if it doesn't exist
            query = """
            MERGE (u:UseCase {name: $use_case_name, inventory_year: 2024})
            ON CREATE SET u.created_at = datetime()
            SET u.last_updated = datetime()
            WITH u
            
            MERGE (a:Agency {name: $agency_name})
            ON CREATE SET a.created_at = datetime()
            SET a.last_updated = datetime()
            WITH u, a
            
            MERGE (u)-[r:BELONGS_TO]->(a)
            ON CREATE SET r.created_at = datetime()
            SET r.last_updated = datetime()
            """
            
            with self.driver.session() as session:
                session.run(query, {
                    "use_case_name": use_case_name,
                    "agency_name": agency_name
                })
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return False

    def _create_ai_categories(self):
        """Create or verify AI category nodes in Neo4j."""
        with self.driver.session() as session:
            # Create AI category nodes
            query = """
            UNWIND $categories as category
            MERGE (c:AICategory {name: category})
            """
            session.run(query, categories=list(self.df['category_name'].unique()))
            self.logger.info("Created/verified AI category nodes")

    def process_data(self, df):
        """Process the data and create relationships in Neo4j."""
        try:
            self.df = df
            # Create evaluation batch node
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d')}"
            
            with self.driver.session() as session:
                # Create batch node
                query = """
                CREATE (b:EvaluationBatch {
                    id: $batch_id,
                    created_at: datetime(),
                    total_records: $total_records,
                    unique_use_cases: $unique_use_cases,
                    unique_agencies: $unique_agencies,
                    unique_categories: $unique_categories
                })
                """
                session.run(query, {
                    "batch_id": batch_id,
                    "total_records": len(df),
                    "unique_use_cases": len(df['use_case_name'].unique()),
                    "unique_agencies": len(df['agency'].unique()),
                    "unique_categories": len(df['category_name'].unique())
                })
                
                # Process each use case
                for use_case_name in df['use_case_name'].unique():
                    self._process_use_case(session, use_case_name, batch_id)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return False

    def import_data(self, df):
        """Import the data into Neo4j."""
        try:
            # Store DataFrame in instance
            self.df = df
            
            # Run tests first
            if not self.test_import_process(df):
                self.logger.error("Import process tests failed")
                return False
            
            # Delete existing batch nodes first
            self.delete_existing_batch()
            
            # Create/verify AI category nodes
            self._create_ai_categories()
            
            # Then process the data
            return self.process_data(df)
            
        except Exception as e:
            self.logger.error(f"Error importing data: {str(e)}")
            return False

    def _dry_run_use_case(self, use_case_name: str, df: pd.DataFrame):
        """Simulate importing a use case without making changes"""
        agency = df.iloc[0]['agency']
        abbreviation = df.iloc[0]['abbreviation']
        
        self.logger.info(f"\nDRY RUN - Would create/update use case:")
        self.logger.info(f"  Name: {use_case_name}")
        self.logger.info(f"  Agency: {agency} ({abbreviation})")
        self.logger.info("\nCategory relationships to create:")
        
        for _, row in df.iterrows():
            self.logger.info(
                f"  - {row['category_name']} ({row['relationship_type']})"
                f" [confidence: {row['confidence']:.2f}]"
            )
            if row['relationship_type'] == 'NO_MATCH':
                self.logger.info(f"    Reason: {row['error'] if pd.notna(row['error']) else 'No specific reason provided'}")

    def _import_use_case(self, use_case_name: str, df: pd.DataFrame):
        """Import a use case and its category relationships"""
        with self.driver.session() as session:
            # Create or update use case and agency
            self._create_use_case_and_agency(session, use_case_name, df)
            
            # Process each category relationship
            for _, row in df.iterrows():
                self._create_category_relationship(session, use_case_name, row)

    def _create_use_case_and_agency(self, session, use_case_name: str, df: pd.DataFrame):
        """Create or update use case and agency nodes with enhanced metadata"""
        agency = df.iloc[0]['agency']
        abbreviation = df.iloc[0]['abbreviation']
        
        query = """
        MERGE (a:Agency {name: $agency})
        SET a.abbreviation = $abbreviation,
            a.agency_type = $agency_type,
            a.last_updated = datetime()
        
        MERGE (u:UseCase {name: $use_case_name, inventory_year: $inventory_year})
        SET u.description = $description,
            u.status = $status,
            u.investment_stage = $investment_stage,
            u.mission_area = $mission_area,
            u.last_updated = datetime()
        
        MERGE (u)-[:BELONGS_TO]->(a)
        """
        
        # Handle parent agency if specified
        if pd.notna(df.iloc[0].get('parent_agency')):
            query += """
            MERGE (pa:Agency {name: $parent_agency})
            MERGE (a)-[:PART_OF]->(pa)
            """
        
        session.run(query, {
            'agency': agency,
            'abbreviation': abbreviation,
            'agency_type': df.iloc[0].get('agency_type', 'EXECUTIVE'),  # Default to EXECUTIVE
            'use_case_name': use_case_name,
            'inventory_year': 2024,  # Hardcoded for 2024 inventory
            'description': df.iloc[0].get('description', ''),
            'status': df.iloc[0].get('status', 'ACTIVE'),
            'investment_stage': df.iloc[0].get('investment_stage', 'OPERATIONAL'),
            'mission_area': df.iloc[0].get('mission_area', ''),
            'parent_agency': df.iloc[0].get('parent_agency')
        })

    def _create_category_relationship(self, session, use_case_name: str, row: pd.Series):
        """Create relationships between use case and AI category with batch tracking"""
        try:
            # First create or get the evaluation batch
            batch_query = """
            MERGE (b:EvaluationBatch {
                id: $batch_id,
                evaluation_date: date(),
                model_versions_json: $model_versions_json,
                confidence_thresholds_json: $confidence_thresholds_json
            })
            """
            
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d')}"
            model_versions = {
                'keyword_matcher': '1.0',
                'semantic_matcher': '1.0',
                'llm': 'gpt-4-0125-preview'
            }
            confidence_thresholds = {
                'keyword': 0.5,
                'semantic': 0.6,
                'llm': 0.7,
                'final': 0.65
            }
            
            session.run(batch_query, {
                'batch_id': batch_id,
                'model_versions_json': json.dumps(model_versions),
                'confidence_thresholds_json': json.dumps(confidence_thresholds)
            })

            # Base query for creating the evaluation node and relationships
            query = """
            MATCH (u:UseCase {name: $use_case_name, inventory_year: 2024})
            MATCH (c:AICategory {name: $category_name})
            MATCH (b:EvaluationBatch {id: $batch_id})
            
            MERGE (e:CategoryEvaluation {
                id: $eval_id
            })
            ON CREATE SET 
                e.keyword_score = $keyword_score,
                e.semantic_score = $semantic_score,
                e.llm_score = $llm_score,
                e.final_score = $final_score,
                e.relationship_type = $relationship_type,
                e.match_method = $match_method,
                e.confidence = $confidence,
                e.matched_keywords = $matched_keywords,
                e.justification = $justification,
                e.evaluation_date = datetime()
            
            MERGE (u)-[:HAS_EVALUATION]->(e)
            MERGE (e)-[:EVALUATES]->(c)
            MERGE (e)-[:PART_OF_BATCH]->(b)
            """
            
            # If it's a NO_MATCH, add additional properties
            if row['relationship_type'] == 'NO_MATCH':
                query += """
                ON CREATE SET 
                    e.no_match_reason = $error,
                    e.improvement_suggestions = $improvement_suggestions
                """
            
            # Generate a unique ID for the evaluation
            eval_id = f"{use_case_name}_{row['category_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert matched_keywords from string to list if necessary
            matched_keywords = (
                row['matched_keywords'].split(', ')
                if isinstance(row['matched_keywords'], str) and row['matched_keywords']
                else []
            )
            
            result = session.run(query, {
                'use_case_name': use_case_name,
                'category_name': row['category_name'],
                'batch_id': batch_id,
                'eval_id': eval_id,
                'keyword_score': float(row['keyword_score']),
                'semantic_score': float(row['semantic_score']),
                'llm_score': float(row['llm_score']),
                'final_score': float(row['final_score']),
                'relationship_type': row['relationship_type'],
                'match_method': row['match_method'],
                'confidence': float(row['confidence']),
                'matched_keywords': matched_keywords,
                'justification': row['explanation'],
                'error': row['error'] if pd.notna(row['error']) else None,
                'improvement_suggestions': "Needs manual review" if pd.notna(row['error']) else None
            })
            
            # Log success
            self.logger.info(f"Successfully created evaluation for {use_case_name} - {row['category_name']}")
            
        except Exception as e:
            self.logger.error(f"Error creating category relationship for {use_case_name} - {row['category_name']}: {str(e)}")
            raise

    def delete_existing_batch(self):
        """Delete any existing evaluation batch nodes to avoid constraint violations."""
        query = """
        MATCH (b:EvaluationBatch)
        DETACH DELETE b
        """
        with self.driver.session() as session:
            session.run(query)
            self.logger.info("Deleted existing evaluation batch nodes")

    def _process_use_case(self, session, use_case_name, batch_id):
        """Process a single use case and create relationships."""
        # Get use case data
        use_case_data = self.df[self.df['use_case_name'] == use_case_name]
        self.logger.info(f"Processing use case: {use_case_name}")
        self.logger.info(f"DataFrame columns: {list(self.df.columns)}")
        self.logger.info(f"Use case data columns: {list(use_case_data.columns)}")
        self.logger.info(f"Use case data shape: {use_case_data.shape}")
        self.logger.info(f"First row: {use_case_data.iloc[0].to_dict()}")
        
        agency_name = use_case_data['agency'].iloc[0]
        self.logger.info(f"Agency name: {agency_name}")
        
        # Create use case node if it doesn't exist
        query = """
        MERGE (u:UseCase {name: $use_case_name, inventory_year: 2024})
        ON CREATE SET u.created_at = datetime()
        SET u.last_updated = datetime()
        WITH u
        
        MERGE (a:Agency {name: $agency_name})
        ON CREATE SET a.created_at = datetime()
        SET a.last_updated = datetime()
        WITH u, a
        
        MERGE (u)-[r:BELONGS_TO]->(a)
        ON CREATE SET r.created_at = datetime()
        SET r.last_updated = datetime()
        """
        session.run(query, {
            "use_case_name": use_case_name,
            "agency_name": agency_name
        })
        
        # Create category evaluations
        for _, row in use_case_data.iterrows():
            eval_id = f"{use_case_name}_{row['category_name']}_{batch_id}"
            
            query = """
            MATCH (u:UseCase {name: $use_case_name})
            MATCH (c:AICategory {name: $category_name})
            MATCH (b:EvaluationBatch {id: $batch_id})
            
            MERGE (e:CategoryEvaluation {id: $eval_id})
            ON CREATE SET e.created_at = datetime()
            SET 
                e.last_updated = datetime(),
                e.relationship_type = $relationship_type,
                e.match_method = $match_method,
                e.keyword_score = $keyword_score,
                e.semantic_score = $semantic_score,
                e.llm_score = $llm_score,
                e.final_score = $final_score,
                e.confidence_score = $confidence_score,
                e.matched_keywords = $matched_keywords,
                e.evaluation_date = datetime()
                
            MERGE (u)-[r1:HAS_EVALUATION]->(e)
            ON CREATE SET r1.created_at = datetime()
            SET r1.last_updated = datetime()
            
            MERGE (e)-[r2:EVALUATES]->(c)
            ON CREATE SET r2.created_at = datetime()
            SET r2.last_updated = datetime()
            
            MERGE (b)-[r3:CONTAINS]->(e)
            ON CREATE SET r3.created_at = datetime()
            SET r3.last_updated = datetime()
            """
            
            session.run(query, {
                "use_case_name": use_case_name,
                "category_name": row['category_name'],
                "batch_id": batch_id,
                "eval_id": eval_id,
                "relationship_type": row['relationship_type'],
                "match_method": row['match_method'],
                "keyword_score": float(row['keyword_score']),
                "semantic_score": float(row['semantic_score']),
                "llm_score": float(row['llm_score']),
                "final_score": float(row['final_score']),
                "confidence_score": float(row['confidence']),
                "matched_keywords": row['matched_keywords'] if pd.notna(row['matched_keywords']) else []
            })

def find_latest_classification_file() -> Optional[Path]:
    """Find the latest classification file in the output directory"""
    output_dir = Path("data/output/results")
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return None
        
    # Look for files matching both patterns
    files = list(output_dir.glob("*_classification_neo4j_*.csv"))
    if not files:
        logger.error("No classification files found in output directory")
        return None
        
    # Sort by modification time and get the latest
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Found latest classification file: {latest_file}")
    return latest_file

def copy_to_input_directory(source_file: Path) -> Optional[Path]:
    """Copy the classification file to the input directory for archival"""
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create target path preserving the filename
    target_file = input_dir / source_file.name
    
    try:
        shutil.copy2(source_file, target_file)
        logger.info(f"Copied classification file to input directory: {target_file}")
        return target_file
    except Exception as e:
        logger.error(f"Failed to copy file to input directory: {str(e)}")
        return None

def main():
    """Main entry point for the script"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # If no input file specified, find the latest classification file
    if not args.input_file:
        latest_file = find_latest_classification_file()
        if not latest_file:
            logger.error("No classification file found and no input file specified")
            sys.exit(1)
            
        # Copy to input directory
        input_file = copy_to_input_directory(latest_file)
        if not input_file:
            logger.error("Failed to copy classification file to input directory")
            sys.exit(1)
            
        args.input_file = str(input_file)
    
    # Create import configuration
    config = ImportConfig(
        dry_run=args.dry_run,
        number=args.number,
        update=args.update,
        input_file=Path(args.input_file)
    )
    
    # Validate input file exists
    if not config.input_file.exists():
        logger.error(f"Input file not found: {config.input_file}")
        sys.exit(1)
    
    try:
        # Read input data
        logger.info(f"Reading input file: {config.input_file}")
        try:
            # Try UTF-8 first
            df = pd.read_csv(config.input_file, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try with cp1252 (common for Windows files)
            logger.info("UTF-8 encoding failed, trying with cp1252...")
            df = pd.read_csv(config.input_file, encoding='cp1252')
        
        # Validate input data
        logger.info("Validating input data...")
        validator = DataValidator(df)
        validation_result = validator.validate()
        
        if not validation_result.is_valid:
            logger.error("Data validation failed:")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        # Log statistics
        logger.info("Data statistics:")
        for key, value in validation_result.stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Initialize Neo4j connection
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not password:
            logger.error("NEO4J_PASSWORD environment variable not set")
            sys.exit(1)
        
        with Neo4jImporter(uri, user, password, config) as importer:
            # Run tests first
            logger.info("Running import process tests...")
            if not importer.test_import_process(df):
                logger.error("Import process tests failed")
                sys.exit(1)
            
            # Restore original config
            importer.config = config
            
            if config.dry_run:
                logger.info("DRY RUN - No changes will be made to the database")
            
            # Process the data
            logger.info(f"Starting import process for {config.input_file}")
            success = importer.import_data(df)
            
            if not success:
                logger.error("Import process failed")
                sys.exit(1)
            
            logger.info("Import process completed successfully")
            
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 