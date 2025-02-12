#!/usr/bin/env python3
"""
Schema Validator

This module validates the Neo4j schema JSON file against expected structure and rules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger('schema_validator')

class SchemaValidator:
    """Validates the Neo4j schema JSON file"""
    
    REQUIRED_TOP_LEVEL_KEYS = ['version', 'description', 'nodes', 'relationships']
    VALID_PROPERTY_TYPES = ['string', 'text', 'datetime', 'boolean', 'float', 'enum', 'array']
    VALID_CARDINALITIES = ['one_to_one', 'one_to_many', 'many_to_one', 'many_to_many']
    
    def __init__(self, schema_path: Path):
        self.schema_path = schema_path
    
    def validate(self) -> bool:
        """
        Validate the schema file
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Load schema
            with open(self.schema_path) as f:
                schema = json.load(f)
            
            # Validate structure
            if not self._validate_structure(schema):
                return False
                
            # Validate nodes
            if not self._validate_nodes(schema['nodes']):
                return False
                
            # Validate relationships
            if not self._validate_relationships(schema['relationships']):
                return False
            
            logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def _validate_structure(self, schema: Dict[str, Any]) -> bool:
        """Validate top-level schema structure"""
        # Check required keys
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in schema:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Validate version format
        if not self._validate_version(schema['version']):
            return False
        
        return True
    
    def _validate_version(self, version: str) -> bool:
        """Validate version string format (x.y.z)"""
        try:
            major, minor, patch = version.split('.')
            int(major), int(minor), int(patch)
            return True
        except ValueError:
            logger.error(f"Invalid version format: {version}. Expected format: x.y.z")
            return False
    
    def _validate_nodes(self, nodes: Dict[str, Any]) -> bool:
        """Validate node definitions"""
        for node_name, node_info in nodes.items():
            # Check required node fields
            if 'properties' not in node_info:
                logger.error(f"Node {node_name} missing properties")
                return False
            
            # Validate properties
            for prop_name, prop_info in node_info['properties'].items():
                if not self._validate_property(node_name, prop_name, prop_info):
                    return False
        
        return True
    
    def _validate_relationships(self, relationships: Dict[str, Any]) -> bool:
        """Validate relationship definitions"""
        for rel_name, rel_info in relationships.items():
            # Check required relationship fields
            required_fields = ['source', 'target', 'cardinality']
            for field in required_fields:
                if field not in rel_info:
                    logger.error(f"Relationship {rel_name} missing {field}")
                    return False
            
            # Validate cardinality
            if rel_info['cardinality'] not in self.VALID_CARDINALITIES:
                logger.error(f"Invalid cardinality for {rel_name}: {rel_info['cardinality']}")
                return False
            
            # Validate properties if present
            if 'properties' in rel_info:
                for prop_name, prop_info in rel_info['properties'].items():
                    if not self._validate_property(rel_name, prop_name, prop_info):
                        return False
        
        return True
    
    def _validate_property(self, parent_name: str, prop_name: str, prop_info: Dict[str, Any]) -> bool:
        """Validate a property definition"""
        # Check property type
        if 'type' not in prop_info:
            logger.error(f"Property {parent_name}.{prop_name} missing type")
            return False
        
        prop_type = prop_info['type']
        if prop_type not in self.VALID_PROPERTY_TYPES:
            logger.error(f"Invalid property type for {parent_name}.{prop_name}: {prop_type}")
            return False
        
        # Validate enum values if type is enum
        if prop_type == 'enum' and 'enum' not in prop_info:
            logger.error(f"Enum property {parent_name}.{prop_name} missing enum values")
            return False
        
        # Validate numeric constraints
        if prop_type == 'float':
            if 'minimum' in prop_info and 'maximum' in prop_info:
                if prop_info['minimum'] > prop_info['maximum']:
                    logger.error(f"Invalid range for {parent_name}.{prop_name}: min > max")
                    return False
        
        return True

def main():
    """CLI entry point for schema validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Neo4j schema')
    parser.add_argument('schema_path', type=Path, help='Path to schema JSON file')
    args = parser.parse_args()
    
    validator = SchemaValidator(args.schema_path)
    if validator.validate():
        logger.info("Schema validation successful!")
    else:
        logger.error("Schema validation failed!")
            
if __name__ == '__main__':
    main() 