#!/usr/bin/env python3
"""
Neo4j Schema Validation Script

This script connects to the Neo4j database and:
1. Extracts the complete schema information
2. Validates our planned changes for the IMPLEMENTS relationship
3. Generates a detailed report of required changes

Usage:
    python validate_neo4j_schema.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neo4j_schema_validator')

@dataclass
class SchemaRelationship:
    """Data structure for relationship metadata"""
    type: str
    properties: Dict[str, str]  # property_name -> data_type
    source_labels: Set[str]
    target_labels: Set[str]
    indexes: List[str]
    constraints: List[str]

@dataclass
class ValidationReport:
    """Data structure for validation results"""
    timestamp: str
    relationship_exists: bool
    property_matches: Dict[str, bool]
    missing_properties: List[str]
    extra_properties: List[str]
    index_status: Dict[str, bool]
    constraint_status: Dict[str, bool]
    recommendations: List[str]

class Neo4jSchemaValidator:
    """Validates Neo4j schema against planned changes"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('neo4j_schema_validator')
    
    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_relationship_schema(self, rel_type: str) -> Optional[SchemaRelationship]:
        """Get complete schema information for a specific relationship type"""
        try:
            with self.driver.session() as session:
                # Get relationship properties using standard Cypher
                prop_query = """
                MATCH ()-[r:IMPLEMENTS]->()
                WITH DISTINCT keys(r) as props
                RETURN props
                """
                props_result = session.run(prop_query).single()
                
                # Get relationship indexes
                index_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE any(label IN labelsOrTypes WHERE label = $rel_type)
                RETURN collect({
                    name: name,
                    type: type,
                    properties: properties
                }) as indexes
                """
                index_result = session.run(index_query, rel_type=rel_type).single()
                
                # Get relationship constraints
                constraint_query = """
                SHOW CONSTRAINTS
                YIELD name, type, labelsOrTypes, properties
                WHERE any(label IN labelsOrTypes WHERE label = $rel_type)
                RETURN collect({
                    name: name,
                    type: type,
                    properties: properties
                }) as constraints
                """
                constraint_result = session.run(constraint_query, rel_type=rel_type).single()
                
                # Get connected node labels
                labels_query = """
                MATCH (a)-[r:IMPLEMENTS]->(b)
                WITH DISTINCT labels(a) as source_labels, labels(b) as target_labels
                RETURN collect(DISTINCT source_labels) as sources, 
                       collect(DISTINCT target_labels) as targets
                LIMIT 1
                """
                labels_result = session.run(labels_query).single()
                
                # Extract relationship metadata
                rel_schema = SchemaRelationship(
                    type=rel_type,
                    properties={},
                    source_labels=set(),
                    target_labels=set(),
                    indexes=[idx["name"] for idx in index_result["indexes"]] if index_result else [],
                    constraints=[con["name"] for con in constraint_result["constraints"]] if constraint_result else []
                )
                
                # Add properties
                if props_result and props_result["props"]:
                    # Get property types
                    for prop in props_result["props"]:
                        type_query = f"""
                        MATCH ()-[r:IMPLEMENTS]->()
                        WHERE r.{prop} IS NOT NULL
                        RETURN type(r.{prop}) as prop_type
                        LIMIT 1
                        """
                        type_result = session.run(type_query).single()
                        if type_result:
                            rel_schema.properties[prop] = type_result["prop_type"]
                
                # Add labels
                if labels_result:
                    for source_label_set in labels_result["sources"]:
                        rel_schema.source_labels.update(source_label_set)
                    for target_label_set in labels_result["targets"]:
                        rel_schema.target_labels.update(target_label_set)
                
                return rel_schema
                
        except Exception as e:
            self.logger.error(f"Failed to get relationship schema: {str(e)}")
            return None

    def validate_planned_changes(self) -> ValidationReport:
        """Validate our planned changes against the actual schema"""
        # Get current schema for IMPLEMENTS relationship
        schema = self.get_relationship_schema("IMPLEMENTS")
        
        # Our planned properties
        planned_properties = {
            "match_method": "string",
            "confidence": "float",
            "final_score": "float",
            "relationship_type": "string",
            "explanation": "string"
        }
        
        # Initialize validation report
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            relationship_exists=bool(schema),
            property_matches={},
            missing_properties=[],
            extra_properties=[],
            index_status={},
            constraint_status={},
            recommendations=[]
        )
        
        if not schema:
            report.recommendations.append(
                "IMPLEMENTS relationship type does not exist. Need to create it with appropriate properties."
            )
            return report
        
        # Validate properties
        for prop, type_hint in planned_properties.items():
            if prop in schema.properties:
                actual_type = schema.properties[prop]
                report.property_matches[prop] = (actual_type.lower() == type_hint.lower())
                if not report.property_matches[prop]:
                    report.recommendations.append(
                        f"Property '{prop}' exists but has type '{actual_type}' instead of '{type_hint}'"
                    )
            else:
                report.missing_properties.append(prop)
                report.recommendations.append(f"Need to add property '{prop}' with type '{type_hint}'")
        
        # Check for extra properties
        for prop in schema.properties:
            if prop not in planned_properties:
                report.extra_properties.append(prop)
                report.recommendations.append(f"Existing property '{prop}' not in planned schema")
        
        # Validate indexes
        expected_indexes = {
            "implements_confidence_idx": "confidence",
            "implements_match_method_idx": "match_method",
            "implements_final_score_idx": "final_score",
            "implements_relationship_type_idx": "relationship_type"
        }
        
        for idx_name, prop in expected_indexes.items():
            exists = any(idx_name in schema.indexes for idx_name in schema.indexes)
            report.index_status[idx_name] = exists
            if not exists:
                report.recommendations.append(f"Need to create index on {prop} property")
        
        # Check label constraints
        if not {"UseCase", "AICategory"}.issubset(schema.source_labels | schema.target_labels):
            report.recommendations.append(
                "Relationship should be between UseCase and AICategory nodes"
            )
        
        return report

    def generate_migration_cypher(self, report: ValidationReport) -> str:
        """Generate Cypher commands for necessary schema changes"""
        commands = []
        
        # Create relationship type if it doesn't exist
        if not report.relationship_exists:
            commands.append("""
            // Create a test relationship to establish the type
            MATCH (u:UseCase), (c:AICategory)
            WITH u LIMIT 1, c LIMIT 1
            CREATE (u)-[r:IMPLEMENTS]->(c)
            DELETE r
            """)
        
        # Add missing properties
        for prop in report.missing_properties:
            commands.append(f"""
            // Add {prop} property to all IMPLEMENTS relationships
            MATCH ()-[r:IMPLEMENTS]->()
            SET r.{prop} = 
                CASE 
                    WHEN r.method IS NOT NULL AND '{prop}' = 'match_method' THEN r.method
                    WHEN '{prop}' IN ['confidence', 'final_score'] THEN 0.0
                    ELSE ''
                END
            """)
        
        # Create missing indexes
        if report.index_status:
            for idx_name, exists in report.index_status.items():
                if not exists:
                    prop = idx_name.replace("implements_", "").replace("_idx", "")
                    commands.append(f"""
                    // Create index for {prop}
                    CREATE INDEX {idx_name} IF NOT EXISTS
                    FOR ()-[r:IMPLEMENTS]-()
                    ON (r.{prop})
                    """)
        
        return "\n".join(commands)

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        return
    
    try:
        # Initialize validator
        with Neo4jSchemaValidator(uri, user, password) as validator:
            # Get validation report
            report = validator.validate_planned_changes()
            
            # Generate output directory
            output_dir = Path("data/output/schema_validation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save validation report
            report_file = output_dir / f"schema_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                # Convert sets to lists for JSON serialization
                report_dict = {
                    "timestamp": report.timestamp,
                    "relationship_exists": report.relationship_exists,
                    "property_matches": report.property_matches,
                    "missing_properties": report.missing_properties,
                    "extra_properties": report.extra_properties,
                    "index_status": report.index_status,
                    "constraint_status": report.constraint_status,
                    "recommendations": report.recommendations
                }
                json.dump(report_dict, f, indent=2)
            
            # Generate migration script if needed
            if report.recommendations:
                migration_file = output_dir / f"migration_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cypher"
                with open(migration_file, 'w') as f:
                    f.write(validator.generate_migration_cypher(report))
            
            # Log results
            logger.info(f"Validation report saved to: {report_file}")
            if report.recommendations:
                logger.info(f"Migration script saved to: {migration_file}")
                logger.info("\nRecommendations:")
                for rec in report.recommendations:
                    logger.info(f"  - {rec}")
            else:
                logger.info("No schema changes required!")
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")

if __name__ == "__main__":
    main() 