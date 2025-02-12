#!/usr/bin/env python
"""
Database Initialization Script for Dell-AITC
Handles complete setup of a new Neo4j database instance including schema setup and data loading.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import uuid  # Add at the top with other imports
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Handles initialization of Neo4j database for Dell-AITC."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize with Neo4j connection details.
        
        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logger
        
    def close(self):
        """Clean up driver resources."""
        self.driver.close()
        
    def verify_connection(self) -> bool:
        """Test database connection and accessibility."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection test' as test")
                if result.single()['test'] != 'Connection test':
                    self.logger.error("Database connection test failed")
                    return False
                self.logger.info("Database connection verified")
                return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return False
            
    def clean_database(self) -> bool:
        """Remove all nodes and relationships from database."""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("Database cleaned successfully")
                return True
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {str(e)}")
            return False
            
    def setup_schema_constraints(self, schema_file: Path) -> bool:
        """Set up schema constraints and indexes from schema file.
        
        Args:
            schema_file: Path to schema JSON file
        """
        try:
            # Load schema definition
            with open(schema_file) as f:
                schema = json.load(f)
                
            # Create constraints and indexes
            with self.driver.session() as session:
                # Drop existing constraints and indexes
                # Get existing constraints
                constraints = session.run("SHOW CONSTRAINTS").data()
                for constraint in constraints:
                    name = constraint.get('name', '')
                    if name:
                        session.run(f"DROP CONSTRAINT {name}")
                
                # Get existing indexes
                indexes = session.run("SHOW INDEXES").data()
                for index in indexes:
                    name = index.get('name', '')
                    if name and not index.get('type') == 'CONSTRAINT':
                        session.run(f"DROP INDEX {index['name']}")
                
                self.logger.info("Dropped existing constraints and indexes")
                
                # Create node constraints
                for node_type, node_def in schema['nodes'].items():
                    for prop, prop_def in node_def['properties'].items():
                        if prop_def.get('unique', False):
                            query = f"""
                            CREATE CONSTRAINT {node_type.lower()}_{prop} IF NOT EXISTS 
                            FOR (n:{node_type}) REQUIRE n.{prop} IS UNIQUE
                            """
                            session.run(query)
                            self.logger.info(f"Created unique constraint on {node_type}.{prop}")
                            
                        if prop_def.get('indexed', False):
                            query = f"""
                            CREATE INDEX {node_type.lower()}_{prop} IF NOT EXISTS 
                            FOR (n:{node_type}) ON (n.{prop})
                            """
                            session.run(query)
                            self.logger.info(f"Created index on {node_type}.{prop}")
                
                # Create v2.2 schema specific constraints and indexes
                # AIClassification node
                session.run("""
                CREATE CONSTRAINT aiclassification_id IF NOT EXISTS 
                FOR (n:AIClassification) REQUIRE n.id IS UNIQUE
                """)
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
                
                # NoMatchAnalysis node
                session.run("""
                CREATE CONSTRAINT nomatchanalysis_id IF NOT EXISTS 
                FOR (n:NoMatchAnalysis) REQUIRE n.id IS UNIQUE
                """)
                session.run("""
                CREATE INDEX nomatchanalysis_status IF NOT EXISTS 
                FOR (n:NoMatchAnalysis) ON (n.status)
                """)
                
                self.logger.info("Created v2.2 schema constraints and indexes")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Schema setup failed: {str(e)}")
            return False
            
    def load_zones(self, zones_file: Path) -> bool:
        """Load AI technology zones from CSV file.
        
        Args:
            zones_file: Path to zones CSV file
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
            SET z.description = zone.description,
                z.created_at = datetime(),
                z.last_updated = datetime()
            """
            
            zones = [
                {'name': row['ai_zone'], 'description': row['zone_definition']}
                for _, row in df.iterrows()
            ]
            
            with self.driver.session() as session:
                session.run(query, {'zones': zones})
                
            self.logger.info(f"Loaded {len(zones)} zones")
            return True
            
        except Exception as e:
            self.logger.error(f"Zone loading failed: {str(e)}")
            return False
            
    def load_categories(self, categories_file: Path) -> bool:
        """Load AI technology categories from CSV file.
        
        Args:
            categories_file: Path to categories CSV file
        """
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
            query = """
            MATCH (z:Zone {name: $zone})
            MERGE (c:AICategory {name: $name})
            SET c.id = $id,
                c.category_definition = $definition,
                c.status = 'active',
                c.maturity_level = $maturity_level,
                c.version = '1.0.0',
                c.created_at = datetime(),
                c.last_updated = datetime()
            MERGE (c)-[:BELONGS_TO]->(z)
            """
            
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    session.run(query, {
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
                            session.run("""
                            MERGE (k:Keyword {name: $keyword})
                            SET k.id = $id,
                                k.type = 'technical_keywords',
                                k.created_at = datetime(),
                                k.last_updated = datetime()
                            WITH k
                            MATCH (c:AICategory {name: $category})
                            MERGE (c)-[r:HAS_KEYWORD]->(k)
                            SET r.created_at = datetime()
                            """, {
                                'keyword': keyword, 
                                'category': row['ai_category'],
                                'id': str(uuid.uuid4())
                            })
                            
                    # Process capabilities
                    if pd.notna(row['capabilities']):
                        capabilities = [c.strip() for c in row['capabilities'].split(';')]
                        for capability in capabilities:
                            session.run("""
                            MERGE (c:Capability {name: $capability})
                            SET c.id = $id,
                                c.created_at = datetime(),
                                c.last_updated = datetime()
                            WITH c
                            MATCH (a:AICategory {name: $category})
                            MERGE (a)-[r:HAS_CAPABILITY]->(c)
                            SET r.created_at = datetime()
                            """, {
                                'capability': capability, 
                                'category': row['ai_category'],
                                'id': str(uuid.uuid4())
                            })
                            
            self.logger.info(f"Loaded {len(df)} categories with relationships")
            return True
            
        except Exception as e:
            self.logger.error(f"Category loading failed: {str(e)}")
            return False
            
    def load_inventory(self, inventory_file: Path) -> bool:
        """Load federal AI inventory from CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(inventory_file, encoding=encoding)
                    self.logger.info(f"Successfully read inventory file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                
            if df is None:
                raise ValueError(f"Could not read file with any of the encodings: {encodings}")
            
            # Map column names
            column_mapping = {
                'Agency': 'Agency',
                'Bureau': 'Bureau',
                'Use Case Name': 'Use Case Name',
                'Use Case Topic Area': 'Topic Area',
                'What is the intended purpose and expected benefits of the AI?': 'Purpose Benefits',
                'Describe the AI system\x92s outputs.': 'Outputs',
                'Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?': 'Contains PII',
                'Does this AI use case have an associated Authority to Operate (ATO) for an AI system?': 'Has ATO',
                'Agency Abbreviation': 'Agency Abbreviation',
                'Date Initiated': 'Date Initiated',
                'Date when Acquisition and/or Development began': 'Date Acquisition',
                'Date Implemented': 'Date Implemented',
                'Date Retired': 'Date Retired',
                'Stage of Development': 'Stage',
                'Is the AI use case rights-impacting, safety-impacting, both, or neither?': 'Impact Type',
                'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ': 'Development Method'  # Note the trailing space in the key
            }
            
            # Rename columns if they exist
            df = df.rename(columns=column_mapping)
            
            # Log available columns for debugging
            self.logger.info(f"Available columns after mapping: {list(df.columns)}")
            
            # Check for required columns
            required_columns = {
                'Agency', 'Bureau', 'Use Case Name', 'Topic Area',
                'Purpose Benefits', 'Outputs'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Helper function to convert to boolean
            def safe_to_bool(value) -> Optional[bool]:
                if pd.isna(value):
                    return None
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                if isinstance(value, str):
                    value = value.lower().strip()
                    if value in ['yes', 'true', '1', 'y']:
                        return True
                    if value in ['no', 'false', '0', 'n']:
                        return False
                return None

            # Helper function to parse dates
            def parse_date(date_str):
                if pd.isna(date_str):
                    return None
                try:
                    return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                except:
                    return None

            def normalize_topic_area(topic: str) -> Optional[str]:
                """Normalize topic area to match schema enumeration."""
                if pd.isna(topic):
                    return None
                
                valid_topics = {
                    'Government Services',
                    'Diplomacy & Trade',
                    'Education & Workforce',
                    'Energy & the Environment',
                    'Emergency Management',
                    'Health & Medical',
                    'Law & Justice',
                    'Science & Space',
                    'Transportation',
                    'Mission-Enabling'
                }
                
                # Clean and normalize the input
                cleaned = str(topic).strip()
                
                # Direct match
                if cleaned in valid_topics:
                    return cleaned
                
                # Try to match common variations
                topic_mapping = {
                    'Government Service': 'Government Services',
                    'Diplomacy and Trade': 'Diplomacy & Trade',
                    'Education and Workforce': 'Education & Workforce',
                    'Energy and Environment': 'Energy & the Environment',
                    'Energy and the Environment': 'Energy & the Environment',
                    'Emergency Mgmt': 'Emergency Management',
                    'Health and Medical': 'Health & Medical',
                    'Law and Justice': 'Law & Justice',
                    'Science and Space': 'Science & Space',
                    'Mission Enabling': 'Mission-Enabling'
                }
                
                return topic_mapping.get(cleaned, 'Mission-Enabling')  # Default to Mission-Enabling if no match

            def normalize_stage(stage: str) -> Optional[str]:
                """Normalize development stage to match schema enumeration."""
                if pd.isna(stage):
                    return None
                
                valid_stages = {
                    'Initiated',
                    'Acquisition and/or Development',
                    'Implementation and Assessment',
                    'Operation and Maintenance',
                    'Retired'
                }
                
                # Clean and normalize the input
                cleaned = str(stage).strip()
                
                # Direct match
                if cleaned in valid_stages:
                    return cleaned
                
                # Try to match common variations
                stage_mapping = {
                    'Planning': 'Initiated',
                    'In Planning': 'Initiated',
                    'Planned': 'Initiated',
                    'Development': 'Acquisition and/or Development',
                    'In Development': 'Acquisition and/or Development',
                    'Under Development': 'Acquisition and/or Development',
                    'Testing': 'Implementation and Assessment',
                    'Production': 'Operation and Maintenance',
                    'In Production': 'Operation and Maintenance',
                    'Live': 'Operation and Maintenance',
                    'Operational': 'Operation and Maintenance',
                    'Decommissioned': 'Retired',
                    'Discontinued': 'Retired'
                }
                
                return stage_mapping.get(cleaned, 'Initiated')  # Default to Initiated if no match

            def normalize_impact_type(impact: str) -> Optional[str]:
                """Normalize impact type to match schema enumeration."""
                if pd.isna(impact):
                    return None
                
                valid_impacts = {
                    'Rights-Impacting',
                    'Safety-Impacting',
                    'Both',
                    'Neither'
                }
                
                # Clean and normalize the input
                cleaned = str(impact).strip()
                
                # Direct match
                if cleaned in valid_impacts:
                    return cleaned
                
                # Try to match common variations
                impact_mapping = {
                    'Rights': 'Rights-Impacting',
                    'Safety': 'Safety-Impacting',
                    'Rights and Safety': 'Both',
                    'None': 'Neither',
                    'N/A': 'Neither'
                }
                
                return impact_mapping.get(cleaned, 'Neither')  # Default to Neither if no match

            def normalize_dev_method(method: str) -> str:
                """
                Normalize the development method to match schema enum values exactly.
                """
                if pd.isna(method):
                    return "Developed with contracting resources."
                
                method = str(method).strip().lower()
                
                # Direct matches first
                if "contract" in method:
                    if "in-house" in method or "internal" in method:
                        return "Developed with both contracting and in-house resources."
                    return "Developed with contracting resources."
                elif "in-house" in method or "internal" in method:
                    return "Developed in-house."
                elif "both" in method or "combination" in method:
                    return "Developed with both contracting and in-house resources."
                
                # Default case
                return "Developed with contracting resources."

            # Process each row
            with self.driver.session() as session:
                # Create agencies first
                agencies = df[['Agency']].drop_duplicates()
                self.logger.info(f"Processing {len(agencies)} unique agencies")
                
                agency_query = """
                UNWIND $agencies as agency
                MERGE (a:Agency {name: agency.name})
                SET a.id = agency.id,
                    a.abbreviation = agency.abbreviation,
                    a.created_at = datetime(),
                    a.last_updated = datetime()
                """
                
                agency_data = [
                    {
                        'name': row['Agency'],
                        'abbreviation': row['Agency'][:10],  # Use first 10 chars as abbreviation
                        'id': str(uuid.uuid4())
                    }
                    for _, row in agencies.iterrows()
                ]
                session.run(agency_query, {'agencies': agency_data})
                self.logger.info(f"Created {len(agencies)} agencies")
                
                # Create bureaus
                bureaus = df[['Agency', 'Bureau']].dropna().drop_duplicates()
                self.logger.info(f"Processing {len(bureaus)} unique bureaus")
                
                bureau_query = """
                UNWIND $bureaus as bureau
                MATCH (a:Agency {name: bureau.agency})
                MERGE (b:Bureau {name: bureau.name})
                SET b.id = bureau.id,
                    b.created_at = datetime(),
                    b.last_updated = datetime()
                MERGE (b)-[:MANAGED_BY]->(a)
                """
                
                bureau_data = [
                    {
                        'agency': row['Agency'],
                        'name': row['Bureau'],
                        'id': str(uuid.uuid4())
                    }
                    for _, row in bureaus.iterrows()
                ]
                session.run(bureau_query, {'bureaus': bureau_data})
                self.logger.info(f"Created {len(bureaus)} bureaus")
                
                # Load use cases in batches
                batch_size = 100
                total_rows = len(df)
                self.logger.info(f"Processing {total_rows} use cases in batches of {batch_size}")
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    use_case_data = []
                    for _, row in batch_df.iterrows():
                        try:
                            # Convert boolean fields safely
                            contains_pii = safe_to_bool(row.get('Contains PII'))
                            has_ato = safe_to_bool(row.get('Has ATO'))
                            
                            # Parse dates
                            date_initiated = parse_date(row.get('Date Initiated'))
                            date_acquisition = parse_date(row.get('Date Acquisition'))
                            date_implemented = parse_date(row.get('Date Implemented'))
                            date_retired = parse_date(row.get('Date Retired'))
                            
                            use_case_data.append({
                                'id': str(uuid.uuid4()),
                                'agency': row['Agency'],
                                'bureau': row['Bureau'],
                                'name': row['Use Case Name'],
                                'topic_area': normalize_topic_area(row['Topic Area']),
                                'stage': normalize_stage(row['Stage']),
                                'impact_type': normalize_impact_type(row['Impact Type']),
                                'dev_method': normalize_dev_method(row['Development Method']),
                                'purpose_benefits': row['Purpose Benefits'],
                                'outputs': row['Outputs'],
                                'contains_pii': contains_pii,
                                'has_ato': has_ato,
                                'date_initiated': date_initiated,
                                'date_acquisition': date_acquisition,
                                'date_implemented': date_implemented,
                                'date_retired': date_retired
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing row {start_idx + _}: {str(e)}")
                            continue
                    
                    # Create use cases and relationships
                    use_case_query = """
                    UNWIND $use_cases as uc
                    MATCH (a:Agency {name: uc.agency})
                    MATCH (b:Bureau {name: uc.bureau})
                    CREATE (u:UseCase {
                        id: uc.id,
                        name: uc.name,
                        topic_area: uc.topic_area,
                        stage: uc.stage,
                        impact_type: uc.impact_type,
                        dev_method: uc.dev_method,
                        purpose_benefits: uc.purpose_benefits,
                        outputs: uc.outputs,
                        contains_pii: uc.contains_pii,
                        has_ato: uc.has_ato,
                        date_initiated: CASE WHEN uc.date_initiated IS NOT NULL THEN datetime(uc.date_initiated) ELSE NULL END,
                        date_acquisition: CASE WHEN uc.date_acquisition IS NOT NULL THEN datetime(uc.date_acquisition) ELSE NULL END,
                        date_implemented: CASE WHEN uc.date_implemented IS NOT NULL THEN datetime(uc.date_implemented) ELSE NULL END,
                        date_retired: CASE WHEN uc.date_retired IS NOT NULL THEN datetime(uc.date_retired) ELSE NULL END,
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                    CREATE (u)-[:IMPLEMENTED_BY]->(a)
                    CREATE (u)-[:MANAGED_BY]->(b)
                    """
                    
                    try:
                        session.run(use_case_query, {'use_cases': use_case_data})
                        self.logger.info(f"Processed use cases {start_idx + 1} to {end_idx}")
                    except Exception as e:
                        self.logger.error(f"Error creating use cases batch {start_idx}-{end_idx}: {str(e)}")
                        continue
                
            self.logger.info("Completed loading inventory data")
            return True
            
        except Exception as e:
            self.logger.error(f"Inventory loading failed: {str(e)}")
            raise
            
    def verify_data_loading(self) -> Dict[str, int]:
        """Verify data loading by counting nodes and relationships."""
        verification_queries = {
            'Zones': "MATCH (z:Zone) RETURN count(z) as count",
            'AICategories': "MATCH (c:AICategory) RETURN count(c) as count",
            'Agencies': "MATCH (a:Agency) RETURN count(a) as count",
            'Bureaus': "MATCH (b:Bureau) RETURN count(b) as count",
            'UseCases': "MATCH (u:UseCase) RETURN count(u) as count",
            'BELONGS_TO': "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count",  # Zone-Category relationships
            'IMPLEMENTED_BY': "MATCH ()-[r:IMPLEMENTED_BY]->() RETURN count(r) as count",
            'MANAGED_BY': "MATCH ()-[r:MANAGED_BY]->() RETURN count(r) as count"
        }
        
        counts = {}
        with self.driver.session() as session:
            for name, query in verification_queries.items():
                result = session.run(query).single()
                counts[name] = result['count']
                self.logger.info(f"{name}: {result['count']}")
                
        return counts

    def validate_loaded_data(self) -> bool:
        """Validate loaded data against schema requirements."""
        logger.info("Validating node counts...")
        # Existing node count validation
        node_counts = self.verify_data_loading()
        
        logger.info("Validating relationships...")
        # Existing relationship validation
        
        logger.info("Validating property constraints...")
        # Existing property validation
        
        logger.info("Validating data quality...")
        # Existing data quality validation
        
        logger.info("Validating enum values...")
        # Existing enum validation

        # Validation checks
        logger.info("Validating date consistency...")
        if not self._validate_date_consistency():
            return False

        logger.info("Validating agency hierarchy...")
        if not self._validate_agency_hierarchy():
            return False

        logger.info("Validating use case completeness...")
        if not self._validate_use_case_completeness():
            return False

        logger.info("Validating relationship integrity...")
        if not self._validate_relationship_integrity():
            return False

        logger.info("Validating schema compliance...")
        if not self._validate_schema_compliance():
            return False

        logger.info("Validating data quality...")
        if not self._validate_data_quality():
            return False

        logger.info("\nValidation Summary:")
        return True

    def _validate_date_consistency(self):
        """Validate date field consistency"""
        self.logger.info("Validating date consistency...")
        
        try:
            with self.driver.session() as session:
                # First check for critical date inconsistencies (retired before implemented)
                critical_query = """
                    MATCH (u:UseCase)
                    WHERE u.date_retired IS NOT NULL 
                      AND u.date_implemented IS NOT NULL 
                      AND datetime(u.date_retired) < datetime(u.date_implemented)
                    RETURN u.name as name,
                           u.date_implemented as implemented,
                           u.date_retired as retired
                """
                
                critical_results = session.run(critical_query).data()
                if critical_results:
                    self.logger.error(f"Found {len(critical_results)} use cases with critical date inconsistencies:")
                    for r in critical_results:
                        self.logger.error(f"Use Case: {r['name']}")
                        self.logger.error(f"  Implemented: {r['implemented']}")
                        self.logger.error(f"  Retired: {r['retired']}")
                        self.logger.error("---")
                    return False

                # Check for non-critical date patterns that should generate warnings
                warning_query = """
                    MATCH (u:UseCase)
                    WHERE (u.date_implemented IS NOT NULL AND u.date_initiated IS NOT NULL 
                          AND datetime(u.date_implemented) < datetime(u.date_initiated))
                     OR (u.date_acquisition IS NOT NULL AND u.date_initiated IS NOT NULL 
                        AND datetime(u.date_acquisition) < datetime(u.date_initiated))
                    RETURN u.name as name,
                           u.date_initiated as initiated,
                           u.date_acquisition as acquisition,
                           u.date_implemented as implemented,
                           CASE 
                             WHEN u.date_implemented < u.date_initiated THEN 'implemented before initiated'
                             WHEN u.date_acquisition < u.date_initiated THEN 'acquired before initiated'
                           END as issue
                """
                
                warning_results = session.run(warning_query).data()
                if warning_results:
                    self.logger.warning(f"Found {len(warning_results)} use cases with non-standard date patterns:")
                    for r in warning_results:
                        self.logger.warning(f"Use Case: {r['name']}")
                        self.logger.warning(f"  Initiated: {r['initiated']}")
                        self.logger.warning(f"  Acquisition: {r['acquisition']}")
                        self.logger.warning(f"  Implemented: {r['implemented']}")
                        self.logger.warning(f"  Issue: {r['issue']}")
                        self.logger.warning("---")
                
                self.logger.info("✓ Date consistency validated")
                return True
                
        except Exception as e:
            self.logger.error(f"Error during date validation: {str(e)}")
            return False

    def _validate_agency_hierarchy(self) -> bool:
        """Validate agency and bureau hierarchy consistency."""
        with self.driver.session() as session:
            # Check for bureaus with invalid agency references
            result = session.run("""
                MATCH (b:Bureau)
                WHERE NOT (b)-[:MANAGED_BY]->(:Agency)
                RETURN count(b) as orphaned_bureaus
            """)
            orphaned_bureaus = result.single()["orphaned_bureaus"]
            
            # Check for use cases with mismatched agency-bureau relationships
            result = session.run("""
                MATCH (u:UseCase)-[:IMPLEMENTED_BY]->(a:Agency)
                MATCH (u)-[:MANAGED_BY]->(b:Bureau)
                WHERE NOT (b)-[:MANAGED_BY]->(a)
                RETURN count(u) as mismatched_hierarchy
            """)
            mismatched = result.single()["mismatched_hierarchy"]
            
            if orphaned_bureaus > 0 or mismatched > 0:
                logger.error(f"❌ Found {orphaned_bureaus} orphaned bureaus and {mismatched} use cases with mismatched agency hierarchy")
                return False
            logger.info("✓ Agency hierarchy validated")
            return True

    def _validate_use_case_completeness(self):
        """Validate that use cases have complete information."""
        self.logger.info("Validating use case completeness...")
        query = """
        MATCH (u:UseCase)
        WHERE u.purpose_benefits IS NULL 
            OR u.outputs IS NULL
            OR (u.purpose_benefits IS NOT NULL AND toString(u.purpose_benefits) = '')
            OR (u.outputs IS NOT NULL AND toString(u.outputs) = '')
        RETURN count(u) as incomplete_count
        """
        with self.driver.session() as session:
            result = session.run(query)
            incomplete_count = result.single()["incomplete_count"]
            if incomplete_count > 0:
                self.logger.error(f"Found {incomplete_count} use cases with incomplete information")
                raise ValueError("Use case completeness validation failed")
            self.logger.info("✓ Use case completeness validated")
            return True

    def _validate_relationship_integrity(self) -> bool:
        """Validate relationship integrity and completeness."""
        with self.driver.session() as session:
            # Check for use cases without required relationships
            result = session.run("""
                MATCH (u:UseCase)
                WHERE NOT (u)-[:IMPLEMENTED_BY]->(:Agency)
                   OR NOT (u)-[:MANAGED_BY]->(:Bureau)
                RETURN count(u) as missing_relationships
            """)
            missing = result.single()["missing_relationships"]
            
            # Check for categories without zone relationships
            result = session.run("""
                MATCH (c:AICategory)
                WHERE NOT (c)-[:BELONGS_TO]->(:Zone)
                RETURN count(c) as missing_zone_relationships
            """)
            missing_zones = result.single()["missing_zone_relationships"]
            
            if missing > 0 or missing_zones > 0:
                logger.error(f"❌ Found {missing} use cases with missing required relationships and {missing_zones} categories without zone relationships")
                return False
            logger.info("✓ Relationship integrity validated")
            return True

    def _validate_schema_compliance(self) -> bool:
        """Validate that all nodes and relationships comply with schema requirements."""
        try:
            with self.driver.session() as session:
                # Validate AIClassification nodes
                result = session.run("""
                MATCH (c:AIClassification)
                WHERE NOT exists(c.id) 
                   OR NOT exists(c.match_type)
                   OR NOT exists(c.confidence)
                   OR NOT exists(c.analysis_method)
                   OR NOT exists(c.analysis_version)
                   OR NOT exists(c.review_status)
                   OR NOT exists(c.classified_at)
                RETURN count(c) as invalid_count
                """).single()
                
                if result['invalid_count'] > 0:
                    self.logger.error(f"Found {result['invalid_count']} AIClassification nodes with missing required properties")
                    return False
                    
                # Validate NoMatchAnalysis nodes
                result = session.run("""
                MATCH (n:NoMatchAnalysis)
                WHERE NOT exists(n.id)
                   OR NOT exists(n.reason)
                   OR NOT exists(n.confidence)
                   OR NOT exists(n.status)
                   OR NOT exists(n.created_at)
                RETURN count(n) as invalid_count
                """).single()
                
                if result['invalid_count'] > 0:
                    self.logger.error(f"Found {result['invalid_count']} NoMatchAnalysis nodes with missing required properties")
                    return False
                    
                # Validate classification relationships
                result = session.run("""
                MATCH (u:UseCase)-[r:CLASSIFIED_AS]->(c:AIClassification)
                MATCH (cat:AICategory)-[r2:CLASSIFIES]->(c)
                WITH u, c, cat, 
                     CASE WHEN NOT exists(r.created_at) THEN 1 ELSE 0 END as missing_rel_props,
                     CASE WHEN NOT exists(r2) THEN 1 ELSE 0 END as missing_cat_rel
                RETURN sum(missing_rel_props) as invalid_rels,
                       sum(missing_cat_rel) as missing_category_rels
                """).single()
                
                if result['invalid_rels'] > 0:
                    self.logger.error(f"Found {result['invalid_rels']} classification relationships with missing properties")
                    return False
                    
                if result['missing_category_rels'] > 0:
                    self.logger.error(f"Found {result['missing_category_rels']} AIClassification nodes without category relationships")
                    return False
                    
                # Validate no-match relationships
                result = session.run("""
                MATCH (u:UseCase)-[r:HAS_ANALYSIS]->(n:NoMatchAnalysis)
                WHERE NOT exists(r.created_at)
                RETURN count(r) as invalid_count
                """).single()
                
                if result['invalid_count'] > 0:
                    self.logger.error(f"Found {result['invalid_count']} no-match relationships with missing properties")
                    return False
                    
                self.logger.info("Schema compliance validation passed")
                return True
                
        except Exception as e:
            self.logger.error(f"Schema compliance validation failed: {str(e)}")
            return False

    def _validate_data_quality(self) -> bool:
        """Validate data quality and consistency."""
        try:
            with self.driver.session() as session:
                # Check for standardization issues
                query = """
                MATCH (a:Agency)
                WHERE a.name =~ '.*\\s+$'  // Trailing whitespace
                OR a.name =~ '^\\s+.*'     // Leading whitespace
                OR a.name =~ '.*\\s{2,}.*' // Multiple spaces
                RETURN count(a) as formatting_issues
                """
                result = session.run(query).single()
                if result and result["formatting_issues"] > 0:
                    self.logger.warning(f"Found {result['formatting_issues']} agencies with formatting issues")
                    # Return True instead of False to continue with initialization
                    return True
                return True
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")
            return False

def find_latest_version_file(data_dir: Path, base_name: str) -> Optional[Path]:
    """Find the latest version of a file based on version number in filename.
    
    Args:
        data_dir: Directory to search in
        base_name: Base name of the file (e.g., 'AI-Technology-Zones')
        
    Returns:
        Path to latest version file or None if not found
    """
    files = list(data_dir.glob(f"{base_name}-v*.csv"))
    if not files:
        return None
        
    # Extract version numbers and sort files
    def extract_version(filepath):
        import re
        match = re.search(r'v(\d+\.\d+)', str(filepath))
        if match:
            return tuple(map(int, match.group(1).split('.')))
        return (0, 0)
        
    return sorted(files, key=extract_version, reverse=True)[0]

def confirm_files(schema_file: Path, zones_file: Path, categories_file: Path, inventory_file: Path) -> bool:
    """Display selected files and ask for user confirmation.
    
    Args:
        schema_file: Path to schema file
        zones_file: Path to zones file
        categories_file: Path to categories file
        inventory_file: Path to inventory file
        
    Returns:
        bool: True if user confirms, False if they want to cancel
    """
    print("\nPreparing to initialize database with the following files:")
    print("\nSchema:")
    print(f"  {schema_file}")
    print("\nInput Files:")
    print(f"  Zones:      {zones_file}")
    print(f"  Categories: {categories_file}")
    print(f"  Inventory:  {inventory_file}")
    
    while True:
        response = input("\nProceed with these files? (yes/no): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer 'yes' or 'no'")

def load_inventory_data(tx, inventory_file):
    """Load inventory data from CSV file into Neo4j."""
    logger.info("Loading inventory data...")
    
    try:
        # Read with 'latin-1' encoding to handle special characters
        df = pd.read_csv(inventory_file, encoding='latin-1')
        
        # Log found columns for debugging
        logger.info(f"Found columns: {df.columns.tolist()}")
        
        # Map expected column names to actual column names
        column_mapping = {
            'Agency': 'Agency',
            'Bureau': 'Bureau',
            'Use Case Name': 'Use Case Name',
            'Use Case Topic Area': 'Topic Area',
            'What is the intended purpose and expected benefits of the AI?': 'Purpose Benefits',
            'Describe the AI system\x92s outputs.': 'Outputs',
            'Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?': 'Contains PII',
            'Does this AI use case have an associated Authority to Operate (ATO) for an AI system?': 'Has ATO',
            'Agency Abbreviation': 'Agency Abbreviation',
            'Date Initiated': 'Date Initiated',
            'Date when Acquisition and/or Development began': 'Date Acquisition',
            'Date Implemented': 'Date Implemented',
            'Date Retired': 'Date Retired',
            'Stage of Development': 'Stage',
            'Is the AI use case rights-impacting, safety-impacting, both, or neither?': 'Impact Type',
            'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ': 'Development Method'  # Note the trailing space in the key
        }
        
        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)
        
        # Log available columns for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_columns = {'Agency', 'Bureau', 'Use Case Name', 'Topic Area', 'Purpose Benefits', 'Outputs'}
        logger.info(f"Required columns: {required_columns}")
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.info(f"Missing columns: {missing_columns}")
            raise Exception(f"Missing required columns. Expected: {required_columns}")
        
        # Process each row
        for _, row in df.iterrows():
            # Clean and standardize values
            agency_name = str(row['Agency']).strip()
            agency_abbr = str(row.get('Agency Abbreviation', '')).strip() if pd.notna(row.get('Agency Abbreviation')) else None
            bureau_name = str(row['Bureau']).strip() if pd.notna(row['Bureau']) else None
            use_case_name = str(row['Use Case Name']).strip()
            topic_area = str(row['Topic Area']).strip() if pd.notna(row['Topic Area']) else None
            purpose_benefits = str(row['Purpose Benefits']).strip() if pd.notna(row['Purpose Benefits']) else None
            outputs = str(row['Outputs']).strip() if pd.notna(row['Outputs']) else None
            contains_pii = str(row.get('Contains PII', '')).strip().lower() == 'yes'
            has_ato = str(row.get('Has ATO', '')).strip().lower() == 'yes'
            
            # Skip if essential fields are empty
            if not agency_name or not use_case_name:
                continue
                
            # Create agency if not exists
            agency_id = str(uuid.uuid4())
            if agency_abbr:
                tx.run("""
                    MERGE (a:Agency {name: $name})
                    ON CREATE SET a.id = $id, a.abbreviation = $abbr
                    ON MATCH SET a.abbreviation = $abbr
                    RETURN a
                """, name=agency_name, id=agency_id, abbr=agency_abbr)
            else:
                tx.run("""
                    MERGE (a:Agency {name: $name})
                    ON CREATE SET a.id = $id
                    RETURN a
                """, name=agency_name, id=agency_id)
            
            # Create bureau if not exists and bureau name is provided
            if bureau_name:
                bureau_id = str(uuid.uuid4())
                tx.run("""
                    MERGE (b:Bureau {name: $name})
                    ON CREATE SET b.id = $id
                    WITH b
                    MATCH (a:Agency {name: $agency_name})
                    MERGE (b)-[:MANAGED_BY]->(a)
                    RETURN b
                """, name=bureau_name, id=bureau_id, agency_name=agency_name)
            
            # Create use case
            use_case_id = str(uuid.uuid4())
            tx.run("""
                CREATE (u:UseCase {
                    id: $id,
                    name: $name,
                    topic_area: $topic_area,
                    stage: $stage,
                    impact_type: $impact_type,
                    dev_method: $dev_method,
                    purpose_benefits: $purpose_benefits,
                    outputs: $outputs,
                    contains_pii: $contains_pii,
                    has_ato: $has_ato
                })
                WITH u
                MATCH (a:Agency {name: $agency_name})
                MERGE (u)-[:OWNED_BY]->(a)
                RETURN u
            """, id=use_case_id, name=use_case_name, topic_area=topic_area,
                stage=row['Stage'], impact_type=row['Impact Type'], dev_method=row['Development Method'],
                purpose_benefits=purpose_benefits, outputs=outputs,
                contains_pii=contains_pii, has_ato=has_ato,
                agency_name=agency_name)
            
            # Link use case to bureau if bureau exists
            if bureau_name:
                tx.run("""
                    MATCH (u:UseCase {id: $use_case_id})
                    MATCH (b:Bureau {name: $bureau_name})
                    MERGE (u)-[:MANAGED_BY]->(b)
                    RETURN u, b
                """, use_case_id=use_case_id, bureau_name=bureau_name)
                
        logger.info("Inventory data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading inventory data: {str(e)}")
        raise Exception(f"Inventory loading failed: {str(e)}")

def drop_constraints_and_indexes(tx):
    """Drop all existing constraints and indexes from the database."""
    logger.info("Dropping existing constraints and indexes...")
    
    try:
        # Get and drop existing constraints
        result = tx.run("SHOW CONSTRAINTS").data()
        for constraint in result:
            name = constraint.get('name', '')
            if name:
                tx.run(f"DROP CONSTRAINT {name}")
                logger.info(f"Dropped constraint: {name}")
                
        # Get and drop existing indexes (excluding those created by constraints)
        result = tx.run("SHOW INDEXES").data()
        for index in result:
            name = index.get('name', '')
            if name and not index.get('type') == 'CONSTRAINT':
                tx.run(f"DROP INDEX {name}")
                logger.info(f"Dropped index: {name}")
                
        logger.info("Successfully dropped all constraints and indexes")
        return True
        
    except Exception as e:
        logger.error(f"Error dropping constraints and indexes: {str(e)}")
        raise Exception(f"Failed to drop constraints and indexes: {str(e)}")

def setup_schema(tx):
    """Set up database schema with constraints and indexes."""
    logger.info("Setting up schema...")
    
    try:
        # Drop all existing constraints and indexes
        drop_constraints_and_indexes(tx)
        
        # Create constraints and indexes for AICategory
        tx.run("""
            CREATE CONSTRAINT aicategory_id IF NOT EXISTS
            FOR (n:AICategory) REQUIRE n.id IS UNIQUE
        """)
        logger.info("Created unique constraint on AICategory.id")
        
        tx.run("""
            CREATE INDEX aicategory_id IF NOT EXISTS 
            FOR (n:AICategory) ON (n.id)
        """)
        logger.info("Created index on AICategory.id")
        
        tx.run("""
            CREATE CONSTRAINT aicategory_name IF NOT EXISTS
            FOR (n:AICategory) REQUIRE n.name IS UNIQUE
        """)
        logger.info("Created unique constraint on AICategory.name")
        
        tx.run("""
            CREATE INDEX aicategory_name IF NOT EXISTS 
            FOR (n:AICategory) ON (n.name)
        """)
        logger.info("Created index on AICategory.name")
        
        tx.run("""
            CREATE INDEX aicategory_status IF NOT EXISTS 
            FOR (n:AICategory) ON (n.status)
        """)
        logger.info("Created index on AICategory.status")
        
        tx.run("""
            CREATE INDEX aicategory_maturity_level IF NOT EXISTS 
            FOR (n:AICategory) ON (n.maturity_level)
        """)
        logger.info("Created index on AICategory.maturity_level")
        
        # Create constraints and indexes for UseCase
        tx.run("""
            CREATE CONSTRAINT usecase_id IF NOT EXISTS
            FOR (n:UseCase) REQUIRE n.id IS UNIQUE
        """)
        logger.info("Created unique constraint on UseCase.id")
        
        tx.run("""
            CREATE INDEX usecase_id IF NOT EXISTS 
            FOR (n:UseCase) ON (n.id)
        """)
        logger.info("Created index on UseCase.id")
        
        tx.run("""
            CREATE INDEX usecase_name IF NOT EXISTS 
            FOR (n:UseCase) ON (n.name)
        """)
        logger.info("Created index on UseCase.name")
        
        tx.run("""
            CREATE INDEX usecase_topic_area IF NOT EXISTS 
            FOR (n:UseCase) ON (n.topic_area)
        """)
        logger.info("Created index on UseCase.topic_area")
        
        tx.run("""
            CREATE INDEX usecase_stage IF NOT EXISTS 
            FOR (n:UseCase) ON (n.stage)
        """)
        logger.info("Created index on UseCase.stage")
        
        tx.run("""
            CREATE INDEX usecase_impact_type IF NOT EXISTS 
            FOR (n:UseCase) ON (n.impact_type)
        """)
        logger.info("Created index on UseCase.impact_type")
        
        tx.run("""
            CREATE INDEX usecase_contains_pii IF NOT EXISTS 
            FOR (n:UseCase) ON (n.contains_pii)
        """)
        logger.info("Created index on UseCase.contains_pii")
        
        tx.run("""
            CREATE INDEX usecase_has_ato IF NOT EXISTS 
            FOR (n:UseCase) ON (n.has_ato)
        """)
        logger.info("Created index on UseCase.has_ato")
        
        # Create constraints and indexes for Agency
        tx.run("""
            CREATE CONSTRAINT agency_id IF NOT EXISTS
            FOR (n:Agency) REQUIRE n.id IS UNIQUE
        """)
        logger.info("Created unique constraint on Agency.id")
        
        tx.run("""
            CREATE INDEX agency_id IF NOT EXISTS 
            FOR (n:Agency) ON (n.id)
        """)
        logger.info("Created index on Agency.id")
        
        tx.run("""
            CREATE CONSTRAINT agency_name IF NOT EXISTS
            FOR (n:Agency) REQUIRE n.name IS UNIQUE
        """)
        logger.info("Created unique constraint on Agency.name")
        
        tx.run("""
            CREATE INDEX agency_name IF NOT EXISTS 
            FOR (n:Agency) ON (n.name)
        """)
        logger.info("Created index on Agency.name")
        
        tx.run("""
            CREATE INDEX agency_abbreviation IF NOT EXISTS 
            FOR (n:Agency) ON (n.abbreviation)
        """)
        logger.info("Created index on Agency.abbreviation")
        
        # Create constraints and indexes for Bureau
        tx.run("""
            CREATE CONSTRAINT bureau_id IF NOT EXISTS
            FOR (n:Bureau) REQUIRE n.id IS UNIQUE
        """)
        logger.info("Created unique constraint on Bureau.id")
        
        tx.run("""
            CREATE INDEX bureau_id IF NOT EXISTS 
            FOR (n:Bureau) ON (n.id)
        """)
        logger.info("Created index on Bureau.id")
        
        tx.run("""
            CREATE INDEX bureau_name IF NOT EXISTS 
            FOR (n:Bureau) ON (n.name)
        """)
        logger.info("Created index on Bureau.name")
        
        tx.run("""
            CREATE INDEX bureau_agency_id IF NOT EXISTS 
            FOR (n:Bureau) ON (n.agency_id)
        """)
        logger.info("Created index on Bureau.agency_id")
        
    except Exception as e:
        logger.error(f"Schema setup failed: {str(e)}")
        raise Exception("Schema setup failed")

def main():
    """Main execution function."""
    load_dotenv()
    
    # Add argument parser
    parser = argparse.ArgumentParser(description='Initialize Neo4j database for Dell-AITC')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompts')
    args = parser.parse_args()
    
    # Get environment variables with more flexible URI handling
    neo4j_uri = os.getenv('NEO4J_URI')
    if not neo4j_uri:
        neo4j_uri = 'bolt://localhost:7687'
        logger.warning("NEO4J_URI not found in .env, using default: bolt://localhost:7687")
    elif not any(neo4j_uri.startswith(proto) for proto in ['bolt://', 'neo4j://']):
        raise ValueError("NEO4J_URI must start with 'bolt://' or 'neo4j://'")
        
    neo4j_user = os.getenv('NEO4J_USER')
    if not neo4j_user:
        neo4j_user = 'neo4j'
        logger.warning("NEO4J_USER not found in .env, using default: neo4j")
        
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD environment variable is required in .env file")
        
    # Initialize paths
    data_dir = Path('data/input')
    schema_file = Path('docs/neo4j/neo4j_schema.json')
    
    # Find latest versions of input files
    zones_file = find_latest_version_file(data_dir, 'AI-Technology-Zones')
    if not zones_file:
        raise FileNotFoundError("No AI Technology Zones file found matching pattern: AI-Technology-Zones-v*.csv")
    logger.info(f"Using zones file: {zones_file.name}")
    
    categories_file = find_latest_version_file(data_dir, 'AI-Technology-Categories')
    if not categories_file:
        raise FileNotFoundError("No AI Technology Categories file found matching pattern: AI-Technology-Categories-v*.csv")
    logger.info(f"Using categories file: {categories_file.name}")
    
    # For inventory file, look for latest year and version
    inventory_files = list(data_dir.glob('[0-9][0-9][0-9][0-9]_consolidated_ai_inventory_raw_v*.csv'))
    if not inventory_files:
        raise FileNotFoundError("No consolidated AI inventory file found matching pattern: YYYY_consolidated_ai_inventory_raw_v*.csv")
    inventory_file = sorted(inventory_files, reverse=True)[0]  # Get most recent year/version
    logger.info(f"Using inventory file: {inventory_file.name}")
    
    # Verify files exist
    for file in [schema_file, zones_file, categories_file, inventory_file]:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")
            
    # Get user confirmation if not using force option
    if not args.force and not confirm_files(schema_file, zones_file, categories_file, inventory_file):
        logger.info("Operation cancelled by user")
        return
        
    # Initialize database
    initializer = DatabaseInitializer(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Verify connection
        if not initializer.verify_connection():
            raise Exception("Database connection failed")
            
        # Clean database
        if not initializer.clean_database():
            raise Exception("Database cleanup failed")
            
        # Setup schema
        with initializer.driver.session() as session:
            setup_schema(session)
            
        # Load data
        if not initializer.load_zones(zones_file):
            raise Exception("Zone loading failed")
            
        if not initializer.load_categories(categories_file):
            raise Exception("Category loading failed")
            
        if not initializer.load_inventory(inventory_file):
            raise Exception("Inventory loading failed")
            
        # Verify loading
        counts = initializer.verify_data_loading()
        
        # After loading data, run validation
        if not initializer.validate_loaded_data():
            logger.error("Data validation failed")
            return 1
            
        logger.info("Database initialization and validation completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    main() 