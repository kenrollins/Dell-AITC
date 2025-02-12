#!/usr/bin/env python3
"""
Neo4j Schema Visualization Generator

This script generates Mermaid diagrams and markdown documentation for the Dell-AITC schema
by reading the schema JSON file and producing visualizations.

Usage:
    python schema_visualization_generator.py

The script will:
1. Read the current schema JSON file
2. Generate Mermaid diagrams for:
   - Core domain structure
   - Relationship properties
   - Classification flow
3. Update the schema visualization markdown file
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('schema_visualization_generator')

@dataclass
class NodeType:
    """Represents a node type in the schema"""
    name: str
    properties: Dict[str, Dict[str, Any]]
    description: Optional[str] = None

    def to_mermaid(self) -> str:
        """Convert node type to Mermaid entity definition"""
        lines = []
        lines.append(f"    {self.name} {{")
        
        for prop_name, prop_info in self.properties.items():
            prop_type = prop_info.get('type', 'string')
            if prop_info.get('format') == 'uuid':
                prop_type = 'uuid'
            
            # Add modifiers with proper spacing
            modifiers = []
            if prop_info.get('unique', False):
                modifiers.append('UK')
            if prop_name == 'id':
                modifiers.append('PK')
            if prop_name.endswith('_id'):
                modifiers.append('FK')
            if prop_info.get('indexed', False):
                modifiers.append('IDX')
            
            # Join modifiers with spaces and wrap in parentheses if there are multiple
            modifier_str = ' '.join(modifiers)
            if len(modifiers) > 1:
                modifier_str = f"({modifier_str})"
            
            if modifier_str:
                lines.append(f"        {prop_type} {prop_name} {modifier_str}")
            else:
                lines.append(f"        {prop_type} {prop_name}")
        
        lines.append("    }")
        return "\n".join(lines)

@dataclass
class RelationType:
    """Represents a relationship type in the schema"""
    name: str
    source: str
    target: str
    cardinality: str
    properties: Dict[str, Dict[str, Any]]
    description: Optional[str] = None

    def to_mermaid_relationship(self) -> str:
        """Convert relationship to Mermaid relationship definition"""
        cardinality_map = {
            'one_to_one': '||--||',
            'one_to_many': '||--|{',
            'many_to_one': '}|--||',
            'many_to_many': '}o--o{'
        }
        rel_symbol = cardinality_map.get(self.cardinality, '}o--o{')
        return f"    {self.source} {rel_symbol} {self.target} : {self.name}"

    def to_mermaid_class(self) -> str:
        """Convert relationship properties to Mermaid class definition"""
        lines = []
        lines.append(f"    class {self.name} {{")
        for prop_name, prop_info in self.properties.items():
            prop_type = prop_info.get('type', 'string')
            lines.append(f"        +{prop_type} {prop_name}")
        lines.append("    }")
        return "\n".join(lines)

class SchemaVisualizer:
    def __init__(self, schema_json_path: Path):
        self.schema_json_path = schema_json_path
        self.schema = self._load_schema()
        self.nodes = self._parse_nodes()
        self.relationships = self._parse_relationships()

    def _load_schema(self) -> Dict:
        """Load schema from JSON file"""
        with open(self.schema_json_path) as f:
            return json.load(f)

    def _parse_nodes(self) -> List[NodeType]:
        """Parse node definitions from schema"""
        nodes = []
        # Add Zone node if not in schema
        if 'Zone' not in self.schema['nodes']:
            nodes.append(NodeType(
                name='Zone',
                properties={
                    'id': {'type': 'string', 'format': 'uuid', 'unique': True, 'indexed': True},
                    'name': {'type': 'string', 'unique': True, 'indexed': True},
                    'description': {'type': 'string'},
                    'created_at': {'type': 'datetime'},
                    'last_updated': {'type': 'datetime'}
                },
                description='Technical zones for categorization'
            ))
        
        # Add Keyword node if not in schema
        if 'Keyword' not in self.schema['nodes']:
            nodes.append(NodeType(
                name='Keyword',
                properties={
                    'id': {'type': 'string', 'format': 'uuid', 'unique': True, 'indexed': True},
                    'name': {'type': 'string', 'unique': True, 'indexed': True},
                    'type': {'type': 'enum', 'enum': ['technical_keywords', 'capabilities', 'business_language']},
                    'relevance_score': {'type': 'float', 'minimum': 0.0, 'maximum': 1.0},
                    'status': {'type': 'enum', 'enum': ['active', 'deprecated']},
                    'created_at': {'type': 'datetime'},
                    'last_updated': {'type': 'datetime'}
                },
                description='Keywords and terms associated with AI categories'
            ))
        
        # Add nodes from schema
        for name, info in self.schema['nodes'].items():
            nodes.append(NodeType(
                name=name,
                properties=info['properties'],
                description=info.get('description')
            ))
        
        return nodes

    def _parse_relationships(self) -> List[RelationType]:
        """Parse relationship definitions from schema"""
        relationships = []
        for name, info in self.schema['relationships'].items():
            relationships.append(RelationType(
                name=name,
                source=info['source'],
                target=info['target'],
                cardinality=info['cardinality'],
                properties=info.get('properties', {}),
                description=info.get('description')
            ))
        return relationships

    def generate_core_structure(self) -> str:
        """Generate Mermaid diagram for core domain structure"""
        lines = []
        lines.append("```mermaid")
        lines.append("erDiagram")
        
        # Add node definitions
        lines.append("    %% Core AI Technology Categories")
        for node in self.nodes:
            if node.name in ['AICategory', 'Zone', 'Keyword']:
                lines.append(node.to_mermaid())
        
        lines.append("\n    %% Federal Use Cases")
        for node in self.nodes:
            if node.name in ['UseCase', 'Agency', 'Bureau']:
                lines.append(node.to_mermaid())
        
        # Add relationships
        lines.append("\n    %% Core Relationships")
        for rel in self.relationships:
            lines.append(rel.to_mermaid_relationship())
        
        lines.append("```")
        return "\n".join(lines)

    def generate_relationship_properties(self) -> str:
        """Generate Mermaid class diagram for relationship properties"""
        lines = []
        lines.append("```mermaid")
        lines.append("classDiagram")
        
        for rel in self.relationships:
            if rel.properties:  # Only show relationships with properties
                lines.append(rel.to_mermaid_class())
        
        lines.append("```")
        return "\n".join(lines)

    def generate_classification_flow(self) -> str:
        """Generate Mermaid sequence diagram for classification flow"""
        return """```mermaid
sequenceDiagram
    participant UC as UseCase
    participant M as Matcher
    participant AC as AICategory
    participant K as Keywords
    
    UC->>M: Extract text content
    M->>K: Extract keywords
    K->>AC: Match categories
    AC->>M: Calculate confidence
    M->>UC: Create USES_TECHNOLOGY
    Note over M,UC: Includes confidence score<br/>and match method
```"""

    def generate_visualization(self) -> str:
        """Generate complete schema visualization markdown"""
        lines = []
        lines.append("# Dell-AITC Schema Visualization v2.1.2\n")
        
        lines.append("## Version Information")
        lines.append("```yaml")
        lines.append("Version: 2.1.2")
        lines.append("Status: Active")
        lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m')}")
        lines.append("```\n")
        
        lines.append("## Core Domain Structure")
        lines.append(self.generate_core_structure())
        lines.append("")
        
        lines.append("## Relationship Properties")
        lines.append(self.generate_relationship_properties())
        lines.append("")
        
        lines.append("## Classification Flow")
        lines.append(self.generate_classification_flow())
        lines.append("")
        
        lines.append("## Legend\n")
        lines.append("### Node Properties")
        lines.append("- PK: Primary Key")
        lines.append("- UK: Unique Key")
        lines.append("- FK: Foreign Key")
        lines.append("- IDX: Indexed\n")
        
        lines.append("### Relationship Types")
        lines.append("- `||--||` : One-to-one")
        lines.append("- `||--|{` : One-to-many (required)")
        lines.append("- `||--o{` : One-to-many (optional)")
        lines.append("- `}|--||` : Many-to-one (required)")
        lines.append("- `}o--||` : Many-to-one (optional)")
        lines.append("- `}o--o{` : Many-to-many (optional)\n")
        
        lines.append("### Property Types")
        lines.append("- uuid: Unique identifier")
        lines.append("- string: Text value")
        lines.append("- text: Long text content")
        lines.append("- enum: Enumerated value")
        lines.append("- boolean: True/False")
        lines.append("- float: Decimal number")
        lines.append("- datetime: Timestamp")
        
        return "\n".join(lines)

def backup_existing_file(file_path: Path) -> None:
    """Create a backup of an existing file"""
    if file_path.exists():
        backup_dir = file_path.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        
        file_path.rename(backup_path)
        logger.info(f"Created backup: {backup_path}")

def main():
    """Main execution function"""
    try:
        # Setup paths
        root_dir = Path(__file__).parent.parent.parent.parent
        schema_json_path = root_dir / 'docs' / 'neo4j' / 'neo4j_schema.json'
        visualization_path = root_dir / 'docs' / 'neo4j' / 'neo4j_schema_visualization.md'
        
        # Create visualizer
        visualizer = SchemaVisualizer(schema_json_path)
        
        # Generate visualization
        visualization_content = visualizer.generate_visualization()
        
        # Backup existing file
        backup_existing_file(visualization_path)
        
        # Write new visualization
        visualization_path.write_text(visualization_content)
        logger.info(f"Generated schema visualization: {visualization_path}")
        
    except Exception as e:
        logger.error(f"Error generating schema visualization: {e}")
        raise

if __name__ == '__main__':
    main() 