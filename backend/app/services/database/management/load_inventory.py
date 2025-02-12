"""
Dell-AITC Federal AI Inventory Loader (v2.2)
Loads federal AI use cases from CSV file into Neo4j database.

Usage:
    python -m backend.app.services.database.management.load_inventory

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
from typing import Optional, Dict, Any

# Configure logging
log_dir = Path("logs/database")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'inventory_load_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
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

# Enum definitions based on data dictionary
VALID_TOPIC_AREAS = [
    "Government Services (includes Benefits and Service Delivery)",
    "Diplomacy & Trade",
    "Education & Workforce",
    "Energy & the Environment",
    "Emergency Management",
    "Health & Medical",
    "Law & Justice",
    "Science & Space",
    "Transportation",
    "Mission-Enabling"
]

VALID_STAGES = [
    "Initiated",
    "Acquisition and/or Development",
    "Implementation and Assessment",
    "Operation and Maintenance",
    "Retired"
]

VALID_IMPACT_TYPES = [
    "Rights-Impacting",
    "Safety-Impacting",
    "Both",
    "Neither"
]

VALID_DEV_METHODS = [
    "Developed with contracting resources.",
    "Developed in-house.",
    "Developed with both contracting and in-house resources."
]

# Add mapping dictionaries after the VALID_ enums
TOPIC_AREA_MAPPINGS = {
    "mission-enabling (internal agency support)": "Mission-Enabling",
    "mission-enabling": "Mission-Enabling",
    "other": "Mission-Enabling",
    "natural language processing": "Mission-Enabling",
    "deep learning": "Mission-Enabling",
    "statistical methods": "Mission-Enabling",
    "classification": "Mission-Enabling",
    "aiml platform/environment": "Mission-Enabling",
    "nlp": "Mission-Enabling",
    "administration of ai governance, processes, and procedures": "Mission-Enabling",
    "department-level ai capabilities and capacity": "Mission-Enabling",
    "ai used in transportation operations": "Transportation",
    "internal dot research project": "Transportation",
    "dot sponsored external research": "Transportation"
}

STAGE_MAPPINGS = {
    "in production": "Operation and Maintenance",
    "in mission": "Operation and Maintenance",
    "planned": "Initiated",
    "ideation": "Initiated",
    "research or administrative action complete": "Implementation and Assessment",
    "research or  administrative action complete": "Implementation and Assessment"  # Note the double space
}

IMPACT_TYPE_MAPPINGS = {
    "safety-impacting": "Safety-Impacting",
    "rights-impacting": "Rights-Impacting",
    "no, use case is too new to fully assess impacts": "Neither",
    "case-by-case assessment": "Neither"
}

DEV_METHOD_MAPPINGS = {
    "no contract/external resources used in development": "Developed in-house.",
    "developed with a combination of in-house and contract/external resources": "Developed with both contracting and in-house resources.",
    "exclusively developed with contract/external resources": "Developed with contracting resources.",
    "data not reported": "Developed with contracting resources."  # Default assumption
}

def clean_string(value: str) -> str:
    """Clean and normalize string values."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace('\n', ' ').replace('\r', '')

class InventoryLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def validate_enum(self, value: str, valid_values: list, mappings: dict = None) -> Optional[str]:
        """Validate and normalize enum values with improved handling."""
        if pd.isna(value):
            return None
        
        # Clean and normalize the input
        clean_value = clean_string(value)
        
        # Handle empty strings and 'none' values
        if not clean_value or clean_value == 'none':
            return None
        
        # Direct match first (case-insensitive)
        for valid_value in valid_values:
            if clean_value == valid_value.lower():
                return valid_value
            
        # Check mappings if provided
        if mappings and clean_value in mappings:
            return mappings[clean_value]
        
        # Handle semicolon-separated values - take the first valid one
        if ';' in clean_value:
            parts = [p.strip() for p in clean_value.split(';')]
            for part in parts:
                if part in mappings:
                    return mappings[part]
                for valid_value in valid_values:
                    if part == valid_value.lower():
                        return valid_value
                    
        # If still no match found, log and return None
        logger.debug(f"No valid enum match found for value: {value} in valid values: {valid_values}")
        return None
        
    def safe_to_bool(self, value) -> Optional[bool]:
        """Convert various string representations to boolean."""
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ['yes', 'true', '1', 'y']:
                return True
            if value in ['no', 'false', '0', 'n']:
                return False
        return None
        
    def parse_date(self, date_str) -> Optional[str]:
        """Parse date string to Neo4j datetime format."""
        if pd.isna(date_str):
            return None
        try:
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except:
            return None
            
    def load_inventory(self, inventory_file: Path) -> bool:
        """Load federal AI use cases from CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(inventory_file, encoding=encoding)
                    logger.info(f"Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any supported encoding")
            
            # Log actual columns for debugging
            logger.info(f"Found columns in file: {list(df.columns)}")
            
            # Map CSV columns to our expected column names
            column_mapping = {
                'Use Case Name': 'name',
                'Agency': 'agency',
                'Agency Abbreviation': 'agency_abbreviation',
                'Bureau': 'bureau',
                'Use Case Topic Area': 'topic_area',
                'What is the intended purpose and expected benefits of the AI?': 'purpose_benefits',
                'Describe the AI system\x92s outputs.': 'outputs',
                'Stage of Development': 'stage',
                'Is the AI use case rights-impacting, safety-impacting, both, or neither?': 'impact_type',
                'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ': 'dev_method',
                'Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?': 'contains_pii',
                'Does this AI use case have an associated Authority to Operate (ATO) for an AI system?': 'has_ato',
                'System Name': 'system_name',
                'Date Initiated': 'date_initiated',
                'Date when Acquisition and/or Development began': 'date_acquisition',
                'Date Implemented': 'date_implemented',
                'Date Retired': 'date_retired'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Validate required columns
            required_columns = {
                'name', 'agency', 'bureau', 'topic_area', 'purpose_benefits', 
                'outputs', 'stage', 'impact_type', 'dev_method', 'contains_pii', 'has_ato'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns after mapping: {missing_columns}")
            
            with self.driver.session() as session:
                # Process each use case
                for _, row in df.iterrows():
                    # Create or update agency with proper validation
                    agency_query = """
                    MERGE (a:Agency {abbreviation: $abbrev})
                    ON CREATE SET 
                        a.id = $id,
                        a.name = $name,
                        a.created_at = datetime()
                    ON MATCH SET 
                        a.name = $name,
                        a.last_updated = datetime()
                    RETURN a
                    """
                    
                    agency_result = session.run(agency_query, {
                        'name': row['agency'],
                        'abbrev': row['agency_abbreviation'],
                        'id': str(uuid.uuid4())
                    }).single()

                    if not agency_result:
                        logger.error(f"Failed to create/update agency: {row['agency']}")
                        continue

                    # Create or update bureau with agency relationship
                    if pd.notna(row['bureau']):
                        bureau_query = """
                        MERGE (b:Bureau {name: $name, agency_id: $agency_id})
                        ON CREATE SET 
                            b.id = $id,
                            b.created_at = datetime()
                        ON MATCH SET 
                            b.last_updated = datetime()
                        WITH b
                        MATCH (a:Agency {name: $agency, abbreviation: $abbrev})
                        MERGE (a)-[r:HAS_BUREAU]->(b)
                        ON CREATE SET r.created_at = datetime()
                        RETURN b
                        """
                        
                        bureau = session.run(bureau_query, {
                            'name': row['bureau'],
                            'agency': row['agency'],
                            'abbrev': row['agency_abbreviation'],
                            'agency_id': agency_result['a']['id'],
                            'id': str(uuid.uuid4())
                        }).single()

                        if not bureau:
                            logger.error(f"Failed to create/update bureau: {row['bureau']}")
                            continue

                    # Validate enums before creating use case
                    validated_topic = self.validate_enum(row['topic_area'], VALID_TOPIC_AREAS, TOPIC_AREA_MAPPINGS)
                    validated_stage = self.validate_enum(row['stage'], VALID_STAGES, STAGE_MAPPINGS)
                    validated_impact = self.validate_enum(row['impact_type'], VALID_IMPACT_TYPES, IMPACT_TYPE_MAPPINGS)
                    validated_dev_method = self.validate_enum(row['dev_method'], VALID_DEV_METHODS, DEV_METHOD_MAPPINGS)

                    if not all([validated_topic, validated_stage, validated_impact, validated_dev_method]):
                        logger.warning(f"Invalid enum values for use case {row['name']}: " +
                                    f"topic={validated_topic}, stage={validated_stage}, " +
                                    f"impact={validated_impact}, dev_method={validated_dev_method}")

                    # Create use case with validated enums
                    usecase_query = """
                    CREATE (u:UseCase {
                        id: $id,
                        name: $name,
                        description: $purpose_benefits,
                        purpose_benefits: $purpose_benefits,
                        outputs: $outputs,
                        topic_area: $topic_area,
                        stage: $stage,
                        impact_type: $impact_type,
                        dev_method: $dev_method,
                        contains_pii: $contains_pii,
                        has_ato: $has_ato,
                        system_name: $system_name,
                        date_initiated: $date_initiated,
                        date_acquisition: $date_acquisition,
                        date_implemented: $date_implemented,
                        date_retired: $date_retired,
                        created_at: datetime(),
                        last_updated: datetime()
                    })
                    WITH u
                    MATCH (a:Agency {name: $agency, abbreviation: $abbrev})
                    MERGE (u)-[r1:IMPLEMENTED_BY]->(a)
                    SET r1.created_at = datetime()
                    """

                    # Add bureau relationship if exists
                    if pd.notna(row['bureau']):
                        usecase_query += """
                        WITH u
                        MATCH (b:Bureau {name: $bureau, agency_id: $agency_id})
                        MERGE (u)-[r2:MANAGED_BY]->(b)
                        SET r2.created_at = datetime()
                        """

                    session.run(usecase_query, {
                        'id': str(uuid.uuid4()),
                        'name': row['name'],
                        'description': row['purpose_benefits'],
                        'purpose_benefits': row['purpose_benefits'],
                        'outputs': row['outputs'],
                        'topic_area': validated_topic,
                        'stage': validated_stage,
                        'impact_type': validated_impact,
                        'dev_method': validated_dev_method,
                        'contains_pii': self.safe_to_bool(row['contains_pii']),
                        'has_ato': self.safe_to_bool(row['has_ato']),
                        'system_name': row.get('system_name'),
                        'date_initiated': self.parse_date(row.get('date_initiated')),
                        'date_acquisition': self.parse_date(row.get('date_acquisition')),
                        'date_implemented': self.parse_date(row.get('date_implemented')),
                        'date_retired': self.parse_date(row.get('date_retired')),
                        'agency': row['agency'],
                        'abbrev': row['agency_abbreviation'],
                        'agency_id': agency_result['a']['id'],
                        'bureau': row['bureau'] if pd.notna(row['bureau']) else None
                    })
            
            logger.info(f"Successfully loaded inventory from {inventory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load inventory: {str(e)}")
            return False

def main():
    """Main entry point for inventory loading."""
    try:
        # Find the inventory file
        data_dir = Path("data/input")
        inventory_file = next(data_dir.glob("[0-9][0-9][0-9][0-9]_consolidated_ai_inventory_raw_v*.csv"), None)
        
        if not inventory_file:
            raise FileNotFoundError("No consolidated AI inventory file found")
            
        logger.info(f"Found inventory file: {inventory_file}")
        
        # Load the inventory
        loader = InventoryLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        success = loader.load_inventory(inventory_file)
        loader.close()
        
        if success:
            logger.info("[SUCCESS] Inventory loading completed successfully")
        else:
            logger.error("[FAILED] Inventory loading failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 