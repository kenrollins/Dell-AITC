#!/usr/bin/env python
"""
Schema Validator for Dell-AITC
Implements comprehensive validation rules for Neo4j schema integrity.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates Neo4j schema structure and relationships."""
    
    REQUIRED_NODE_PROPERTIES = {
        'AICategory': {
            'id', 'name', 'category_definition', 'status',
            'maturity_level', 'zone_id', 'created_at',
            'last_updated', 'version'
        },
        'Zone': {
            'id', 'name', 'description', 'status',
            'created_at', 'last_updated'
        },
        'Keyword': {
            'id', 'name', 'type', 'relevance_score',
            'status', 'created_at', 'last_updated'
        },
        'Capability': {
            'id', 'name', 'description', 'type',
            'status', 'created_at', 'last_updated'
        }
    }
    
    VALID_KEYWORD_TYPES = {
        'technical_keywords',
        'capabilities',
        'business_language'
    }
    
    def __init__(self, schema_path: Path):
        """Initialize the schema validator.
        
        Args:
            schema_path: Path to the schema JSON file
        """
        self.schema_path = schema_path
        self.validation_errors: List[str] = []
        
    def load_schema(self) -> Optional[Dict]:
        """Load and parse the schema JSON file."""
        try:
            with open(self.schema_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return None
            
    def validate_node_properties(self, schema: Dict) -> bool:
        """Validate required properties for each node type."""
        valid = True
        
        for node_type, required_props in self.REQUIRED_NODE_PROPERTIES.items():
            if node_type not in schema.get('nodes', {}):
                self.validation_errors.append(f"Missing node type: {node_type}")
                valid = False
                continue
                
            node_props = set(schema['nodes'][node_type].get('properties', {}).keys())
            missing_props = required_props - node_props
            
            if missing_props:
                self.validation_errors.append(
                    f"Missing required properties for {node_type}: {missing_props}"
                )
                valid = False
                
        return valid
        
    def validate_relationships(self, schema: Dict) -> bool:
        """Validate relationship definitions and constraints."""
        valid = True
        relationships = schema.get('relationships', {})
        
        for rel_type, rel_def in relationships.items():
            if not all(key in rel_def for key in ['source', 'target', 'properties']):
                self.validation_errors.append(
                    f"Invalid relationship definition for {rel_type}"
                )
                valid = False
                
        return valid
        
    def validate_keyword_types(self, schema: Dict) -> bool:
        """Validate keyword type constraints."""
        valid = True
        
        if 'Keyword' in schema.get('nodes', {}):
            type_prop = schema['nodes']['Keyword'].get('properties', {}).get('type', {})
            if not type_prop.get('enum') == list(self.VALID_KEYWORD_TYPES):
                self.validation_errors.append(
                    f"Invalid keyword type enum. Expected: {self.VALID_KEYWORD_TYPES}"
                )
                valid = False
                
        return valid
        
    def validate_version_metadata(self, schema: Dict) -> bool:
        """Validate version metadata across all nodes."""
        valid = True
        required_metadata = {'created_at', 'last_updated', 'version'}
        
        for node_type, node_def in schema.get('nodes', {}).items():
            node_props = set(node_def.get('properties', {}).keys())
            missing_metadata = required_metadata - node_props
            
            if missing_metadata:
                self.validation_errors.append(
                    f"Missing version metadata for {node_type}: {missing_metadata}"
                )
                valid = False
                
        return valid
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        schema = self.load_schema()
        if not schema:
            return False
            
        validations = [
            self.validate_node_properties(schema),
            self.validate_relationships(schema),
            self.validate_keyword_types(schema),
            self.validate_version_metadata(schema)
        ]
        
        return all(validations)
        
    def get_validation_errors(self) -> List[str]:
        """Return list of validation errors."""
        return self.validation_errors

def main():
    """CLI entry point for schema validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Neo4j schema')
    parser.add_argument('schema_path', type=Path, help='Path to schema JSON file')
    args = parser.parse_args()
    
    validator = SchemaValidator(args.schema_path)
    if validator.validate_all():
        logger.info("Schema validation successful!")
    else:
        logger.error("Schema validation failed!")
        for error in validator.get_validation_errors():
            logger.error(error)
            
if __name__ == '__main__':
    main() 