#!/usr/bin/env python3
"""
Schema Documentation Update Script

This script orchestrates the process of updating all schema documentation files
in the correct order, ensuring consistency across all files.

Usage:
    python update_schema_docs.py [--no-backup]

The script will:
1. Validate the schema JSON
2. Generate schema documentation
3. Generate schema visualization
4. Validate all files are consistent
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('schema_docs_updater')

class SchemaDocsUpdater:
    def __init__(self, root_dir: Path, create_backups: bool = True):
        self.root_dir = root_dir
        self.create_backups = create_backups
        self.docs_dir = root_dir / 'docs' / 'neo4j'
        self.schema_json_path = self.docs_dir / 'neo4j_schema.json'
        self.schema_doc_path = self.docs_dir / 'neo4j_schema_documentation.md'
        self.schema_viz_path = self.docs_dir / 'neo4j_schema_visualization.md'
        
        # Import related modules
        try:
            # Add script directory to path for imports
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.append(str(script_dir))
            
            # Direct imports instead of relative imports
            import schema_validator
            import schema_visualization_generator
            self.validator = schema_validator.SchemaValidator(self.schema_json_path)
            self.visualizer = schema_visualization_generator.SchemaVisualizer(self.schema_json_path)
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Make sure schema_validator.py and schema_visualization_generator.py are in the same directory")
            raise

    def backup_file(self, file_path: Path) -> Optional[Path]:
        """Create a backup of a file if backups are enabled"""
        if not self.create_backups or not file_path.exists():
            return None
            
        backup_dir = self.docs_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path

    def validate_schema_json(self) -> bool:
        """Validate the schema JSON file"""
        logger.info("Validating schema JSON...")
        return self.validator.validate()

    def generate_documentation(self) -> bool:
        """Generate schema documentation markdown"""
        logger.info("Generating schema documentation...")
        try:
            # Load schema
            with open(self.schema_json_path) as f:
                schema = json.load(f)
            
            # Backup existing documentation
            self.backup_file(self.schema_doc_path)
            
            # Generate documentation content
            lines = []
            
            # Header
            lines.append(f"# Dell-AITC Neo4j Schema v{schema['version']}\n")
            
            # Overview
            lines.append("## Overview")
            lines.append(schema.get('description', 'Schema definition for the Dell-AITC AI Technology Categorization system.'))
            lines.append("")
            
            # Version Information
            lines.append("## Version Information")
            lines.append("```yaml")
            lines.append(f"Version: {schema['version']}")
            lines.append("Status: Released")
            lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m')}")
            lines.append("```\n")
            
            # Domain Model
            lines.append("## Domain Model\n")
            lines.append("### Core Domains")
            lines.append("1. AI Technology Classification")
            lines.append("   - AI Categories")
            lines.append("   - Technical Zones")
            lines.append("   - Keywords and Capabilities")
            lines.append("2. Federal Use Cases")
            lines.append("   - Use Case Details")
            lines.append("   - Agency Structure")
            lines.append("   - Implementation Status")
            lines.append("3. Technology Mapping")
            lines.append("   - Category Matching")
            lines.append("   - Confidence Scoring")
            lines.append("   - Validation Process\n")
            
            # Node Types
            lines.append("## Node Types\n")
            for node_name, node_info in schema['nodes'].items():
                lines.append(f"### {node_name}")
                if node_info.get('description'):
                    lines.append(node_info['description'])
                
                lines.append("\n```yaml")
                lines.append("Properties:")
                for prop_name, prop_info in node_info['properties'].items():
                    prop_type = prop_info.get('type', 'string')
                    if prop_info.get('format') == 'uuid':
                        prop_type = 'uuid'
                    
                    description = prop_info.get('description', '')
                    if description:
                        lines.append(f"  {prop_name}: {prop_type}".ljust(30) + f"# {description}")
                    else:
                        lines.append(f"  {prop_name}: {prop_type}")
                
                lines.append("\nConstraints:")
                # Add unique constraints
                for prop_name, prop_info in node_info['properties'].items():
                    if prop_info.get('unique', False):
                        lines.append(f"  - {prop_name} must be unique")
                
                # Add enum constraints
                for prop_name, prop_info in node_info['properties'].items():
                    if prop_info.get('enum'):
                        lines.append(f"  - valid {prop_name} values: {', '.join(prop_info['enum'])}")
                
                # Add other constraints
                for prop_name, prop_info in node_info['properties'].items():
                    if 'minimum' in prop_info or 'maximum' in prop_info:
                        min_val = prop_info.get('minimum', '')
                        max_val = prop_info.get('maximum', '')
                        lines.append(f"  - {prop_name} range: {min_val} to {max_val}")
                    
                    if prop_info.get('min_length'):
                        lines.append(f"  - {prop_name} minimum length: {prop_info['min_length']} chars")
                
                lines.append("```\n")
            
            # Relationships
            lines.append("## Relationships\n")
            for rel_name, rel_info in schema['relationships'].items():
                lines.append(f"### {rel_name}")
                if rel_info.get('description'):
                    lines.append(rel_info['description'])
                
                lines.append("\n```yaml")
                lines.append(f"Source: {rel_info['source']}")
                lines.append(f"Target: {rel_info['target']}")
                lines.append(f"Cardinality: {rel_info['cardinality'].upper()}")
                
                if rel_info.get('properties'):
                    lines.append("\nProperties:")
                    for prop_name, prop_info in rel_info['properties'].items():
                        prop_type = prop_info.get('type', 'string')
                        description = prop_info.get('description', '')
                        if description:
                            lines.append(f"  {prop_name}: {prop_type}".ljust(30) + f"# {description}")
                        else:
                            lines.append(f"  {prop_name}: {prop_type}")
                
                if any(p.get('enum') or p.get('minimum') or p.get('maximum') 
                      for p in rel_info.get('properties', {}).values()):
                    lines.append("\nConstraints:")
                    for prop_name, prop_info in rel_info['properties'].items():
                        if prop_info.get('enum'):
                            lines.append(f"  - valid {prop_name} values: {', '.join(prop_info['enum'])}")
                        if 'minimum' in prop_info or 'maximum' in prop_info:
                            min_val = prop_info.get('minimum', '')
                            max_val = prop_info.get('maximum', '')
                            lines.append(f"  - {prop_name} range: {min_val} to {max_val}")
                
                lines.append("```\n")
            
            # Indexes
            lines.append("## Indexes\n")
            lines.append("### Node Indexes")
            lines.append("```cypher")
            # Generate index statements for each node type
            for node_name, node_info in schema['nodes'].items():
                indexed_props = [
                    prop_name for prop_name, prop_info in node_info['properties'].items()
                    if prop_info.get('indexed', False)
                ]
                for prop in indexed_props:
                    lines.append(f"CREATE INDEX ON :{node_name}({prop})")
            lines.append("```\n")
            
            # Fulltext Indexes
            lines.append("### Fulltext Indexes")
            lines.append("```cypher")
            lines.append("// Text search indexes")
            lines.append("CREATE FULLTEXT INDEX ON :UseCase([purpose_benefits, outputs])")
            lines.append("CREATE FULLTEXT INDEX ON :AICategory([category_definition])")
            lines.append("```\n")
            
            # Data Quality Rules
            lines.append("## Data Quality Rules\n")
            lines.append("### Node Rules")
            for node_name, node_info in schema['nodes'].items():
                lines.append(f"1. {node_name}")
                if node_info.get('description'):
                    lines.append(f"   - {node_info['description']}")
                for prop_name, prop_info in node_info['properties'].items():
                    if prop_info.get('description'):
                        lines.append(f"   - {prop_info['description']}")
                lines.append("")
            
            lines.append("### Relationship Rules")
            lines.append("1. USES_TECHNOLOGY")
            lines.append("   - Must have confidence score")
            lines.append("   - Must specify match method")
            lines.append("   - Should be validated for high-impact cases\n")
            
            lines.append("2. General")
            lines.append("   - No orphaned nodes")
            lines.append("   - Required timestamps")
            lines.append("   - Valid property ranges\n")
            
            # Query Optimization
            lines.append("## Query Optimization")
            lines.append("1. Use case text search via fulltext indexes")
            lines.append("2. Category matching via keyword and semantic indexes")
            lines.append("3. Agency hierarchy traversal optimization")
            lines.append("4. Impact analysis queries\n")
            
            # Implementation Notes
            lines.append("## Implementation Notes")
            lines.append("1. Use UUIDs for all IDs")
            lines.append("2. Maintain audit timestamps")
            lines.append("3. Validate all enums")
            lines.append("4. Check referential integrity")
            
            # Write the documentation
            self.schema_doc_path.write_text('\n'.join(lines))
            logger.info("Schema documentation generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return False

    def generate_visualization(self) -> bool:
        """Generate schema visualization markdown"""
        logger.info("Generating schema visualization...")
        try:
            # Backup existing visualization
            self.backup_file(self.schema_viz_path)
            
            # Generate new visualization using the existing visualizer
            visualization_content = self.visualizer.generate_visualization()
            
            # Write the visualization to file
            with open(self.schema_viz_path, 'w') as f:
                f.write(visualization_content)
            
            logger.info("Schema visualization generated successfully")
            return True
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return False

    def validate_consistency(self) -> bool:
        """Validate consistency across all schema files"""
        logger.info("Validating consistency across schema files...")
        try:
            # Load schema version from each file
            schema_version = None
            
            # Check JSON schema version
            with open(self.schema_json_path) as f:
                schema = json.load(f)
                schema_version = schema.get('version')
            
            # Check documentation version
            with open(self.schema_doc_path) as f:
                doc_content = f.read()
                if schema_version not in doc_content:
                    logger.error("Schema version mismatch in documentation")
                    return False
            
            # Check visualization version
            with open(self.schema_viz_path) as f:
                viz_content = f.read()
                if schema_version not in viz_content:
                    logger.error("Schema version mismatch in visualization")
                    return False
            
            logger.info("All schema files are consistent")
            return True
        except Exception as e:
            logger.error(f"Error validating consistency: {e}")
            return False

    def update_all(self) -> bool:
        """Run the complete update process"""
        steps = [
            (self.validate_schema_json, "Schema JSON validation"),
            (self.generate_documentation, "Documentation generation"),
            (self.generate_visualization, "Visualization generation"),
            (self.validate_consistency, "Consistency validation")
        ]
        
        for step_func, step_name in steps:
            logger.info(f"Starting {step_name}...")
            if not step_func():
                logger.error(f"Failed at {step_name}")
                return False
            logger.info(f"Completed {step_name}")
        
        logger.info("Schema documentation update completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Update schema documentation files")
    parser.add_argument('--no-backup', action='store_true', help="Skip creating backups")
    args = parser.parse_args()
    
    try:
        root_dir = Path(__file__).parent.parent.parent.parent
        updater = SchemaDocsUpdater(root_dir, create_backups=not args.no_backup)
        
        if updater.update_all():
            logger.info("Schema documentation update completed successfully")
        else:
            logger.error("Schema documentation update failed")
            exit(1)
    except Exception as e:
        logger.error(f"Error updating schema documentation: {e}")
        raise

if __name__ == '__main__':
    main() 