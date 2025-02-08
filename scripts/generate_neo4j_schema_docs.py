#!/usr/bin/env python3
"""
Neo4j Schema Documentation Generator

This script connects to the Neo4j database, extracts the complete schema information,
and generates updated documentation files in both Markdown and JSON formats.

Features:
- Extracts node labels, properties, and constraints
- Documents relationship types and their properties
- Captures indexes and constraints
- Backs up existing documentation
- Generates both human-readable markdown and machine-readable JSON

Usage:
    python generate_neo4j_schema_docs.py
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from neo4j import GraphDatabase, Driver
from neo4j.time import DateTime
from dotenv import load_dotenv
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neo4j_schema_generator')

class SchemaEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle sets and other special types"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, DateTime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class NodeLabel:
    """Represents a Neo4j node label and its properties"""
    label: str
    properties: Dict[str, str]
    relationships_out: Set[str] = None  # Relationship types going out
    relationships_in: Set[str] = None   # Relationship types coming in

    def __post_init__(self):
        if self.relationships_out is None:
            self.relationships_out = set()
        if self.relationships_in is None:
            self.relationships_in = set()

    def to_dict(self):
        """Convert to dictionary with lists instead of sets"""
        return {
            'label': self.label,
            'properties': self.properties,
            'relationships_out': list(self.relationships_out),
            'relationships_in': list(self.relationships_in)
        }

@dataclass
class RelationType:
    """Represents a Neo4j relationship type and its properties"""
    type: str
    properties: Dict[str, str]
    source_labels: Set[str] = None  # Labels of source nodes
    target_labels: Set[str] = None  # Labels of target nodes

    def __post_init__(self):
        if self.source_labels is None:
            self.source_labels = set()
        if self.target_labels is None:
            self.target_labels = set()

    def to_dict(self):
        """Convert to dictionary with lists instead of sets"""
        return {
            'type': self.type,
            'properties': self.properties,
            'source_labels': list(self.source_labels),
            'target_labels': list(self.target_labels)
        }

@dataclass
class DatabaseSchema:
    """Represents the complete Neo4j database schema"""
    node_labels: List[NodeLabel]
    relationship_types: List[RelationType]
    timestamp: str

    def to_dict(self):
        """Convert to dictionary with proper serialization"""
        return {
            'node_labels': [node.to_dict() for node in self.node_labels],
            'relationship_types': [rel.to_dict() for rel in self.relationship_types],
            'timestamp': self.timestamp
        }

class SchemaExtractor:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def _get_property_type(self, value: Any) -> str:
        """Get the type of a property value"""
        if value is None:
            return 'null'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'list'
        elif isinstance(value, dict):
            return 'map'
        else:
            return 'unknown'

    def _get_node_labels(self) -> List[NodeLabel]:
        with self.driver.session() as session:
            # First try using APOC if available
            try:
                result = session.run("""
                    CALL apoc.meta.schema()
                    YIELD value
                    RETURN value
                """)
                schema_data = result.single()
                if schema_data:
                    node_labels = []
                    for label, data in schema_data['value'].items():
                        if data.get('type') == 'node':
                            properties = {
                                prop: prop_data.get('type', 'unknown')
                                for prop, prop_data in data.get('properties', {}).items()
                            }
                            node_labels.append(NodeLabel(label=label, properties=properties))
                    return node_labels
            except Exception as e:
                logger.warning(f"APOC not available, falling back to manual schema extraction: {e}")

            # First get all labels
            labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels = labels_result.single()["labels"]
            
            node_labels = []
            
            # Then process each label
            for label in labels:
                # Get node properties and relationships
                result = session.run("""
                    MATCH (n:%s)
                    WITH n LIMIT 1
                    WITH n,
                         CASE WHEN n IS NULL THEN {} ELSE properties(n) END as props,
                         [(n)-[r]->() | type(r)] as out_rels,
                         [(n)<-[r]-() | type(r)] as in_rels
                    RETURN {
                        label: $label,
                        properties: props,
                        relationships_out: out_rels,
                        relationships_in: in_rels
                    } as schema
                """ % label, label=label)
                
                record = result.single()
                if record:
                    node_labels.append(
                        NodeLabel(
                            label=record['schema']['label'],
                            properties={
                                k: self._get_property_type(v)
                                for k, v in record['schema']['properties'].items()
                            },
                            relationships_out=set(record['schema']['relationships_out']),
                            relationships_in=set(record['schema']['relationships_in'])
                        )
                    )
            
            return node_labels

    def _get_relationship_types(self) -> List[RelationType]:
        with self.driver.session() as session:
            # First try using APOC if available
            try:
                result = session.run("""
                    CALL apoc.meta.schema()
                    YIELD value
                    RETURN value
                """)
                schema_data = result.single()
                if schema_data:
                    rel_types = []
                    for rel_type, data in schema_data['value'].items():
                        if data.get('type') == 'relationship':
                            properties = {
                                prop: prop_data.get('type', 'unknown')
                                for prop, prop_data in data.get('properties', {}).items()
                            }
                            rel_types.append(RelationType(type=rel_type, properties=properties))
                    return rel_types
            except Exception as e:
                logger.warning(f"APOC not available, falling back to manual schema extraction: {e}")

            # First get all relationship types
            types_result = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
            )
            rel_types = types_result.single()["types"]
            
            relationship_types = []
            
            # Then process each relationship type
            for rel_type in rel_types:
                # Get relationship properties and connected node labels
                result = session.run("""
                    MATCH (a)-[r:%s]->(b)
                    WITH r, a, b LIMIT 1
                    WITH r,
                         CASE WHEN r IS NULL THEN {} ELSE properties(r) END as props,
                         CASE WHEN a IS NOT NULL THEN labels(a) ELSE [] END as source_labels,
                         CASE WHEN b IS NOT NULL THEN labels(b) ELSE [] END as target_labels
                    RETURN {
                        type: $rel_type,
                        properties: props,
                        source_labels: source_labels,
                        target_labels: target_labels
                    } as schema
                """ % rel_type, rel_type=rel_type)
                
                record = result.single()
                if record:
                    relationship_types.append(
                        RelationType(
                            type=record['schema']['type'],
                            properties={
                                k: self._get_property_type(v)
                                for k, v in record['schema']['properties'].items()
                            },
                            source_labels=set(record['schema']['source_labels'] or []),
                            target_labels=set(record['schema']['target_labels'] or [])
                        )
                    )
            
            return relationship_types

    def get_schema(self) -> DatabaseSchema:
        """Extract complete schema information from the database"""
        try:
            node_labels = self._get_node_labels()
            relationship_types = self._get_relationship_types()
            
            return DatabaseSchema(
                node_labels=node_labels,
                relationship_types=relationship_types,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Failed to extract schema: {e}")
            raise

def generate_markdown(schema: DatabaseSchema) -> str:
    """Generate markdown documentation from the schema"""
    lines = [
        "# Neo4j Database Schema Documentation",
        f"\nGenerated at: {schema.timestamp}\n",
        "## Node Labels",
        "\nThe following node labels are defined in the database:\n"
    ]

    for node in schema.node_labels:
        lines.append(f"### :{node.label}")
        if node.properties:
            lines.append("\nProperties:")
            lines.append("| Property | Type |")
            lines.append("|----------|------|")
            for prop, prop_type in node.properties.items():
                lines.append(f"| {prop} | {prop_type} |")
        else:
            lines.append("\nNo properties defined.")
        lines.append("")

    lines.extend([
        "## Relationship Types",
        "\nThe following relationship types are defined in the database:\n"
    ])

    for rel in schema.relationship_types:
        lines.append(f"### :{rel.type}")
        if rel.properties:
            lines.append("\nProperties:")
            lines.append("| Property | Type |")
            lines.append("|----------|------|")
            for prop, prop_type in rel.properties.items():
                lines.append(f"| {prop} | {prop_type} |")
        else:
            lines.append("\nNo properties defined.")
        lines.append("")

    return "\n".join(lines)

def generate_mermaid_diagram(schema: DatabaseSchema) -> str:
    """Generate a Mermaid ER diagram representation of the schema"""
    lines = [
        "```mermaid",
        "erDiagram",
        "    %% Core AI Technology Categories",
    ]
    
    # Track relationships to avoid duplicates
    added_relationships = set()
    
    # Group entities by their role in the schema
    core_entities = {"AICategory", "Capability", "Keyword", "IntegrationPattern", "Zone"}
    use_case_entities = {"UseCase", "Agency", "Bureau", "System", "PurposeBenefit", "Output"}
    metadata_entities = {"SchemaMetadata", "Version", "Metadata"}
    
    # Helper function to format entity with properties
    def format_entity(node):
        # Start entity definition
        lines.append(f"    {node.label} {{")
        # Add properties with types
        for prop, prop_type in sorted(node.properties.items()):
            # Convert Neo4j types to SQL-like types for better readability
            type_str = {
                'string': 'string',
                'integer': 'int',
                'float': 'float',
                'boolean': 'boolean',
                'datetime': 'datetime',
                'list': 'array',
                'map': 'json',
                'unknown': 'any'
            }.get(prop_type.lower(), prop_type)
            lines.append(f"        {type_str} {prop}")
        lines.append("    }")
        lines.append("")
    
    # Add core entities first
    lines.append("    %% Core AI Technology Categories")
    for node in schema.node_labels:
        if node.label in core_entities:
            format_entity(node)
    
    # Add use case entities
    lines.append("\n    %% Federal Use Case Structure")
    for node in schema.node_labels:
        if node.label in use_case_entities:
            format_entity(node)
    
    # Add metadata entities
    lines.append("\n    %% Metadata and Versioning")
    for node in schema.node_labels:
        if node.label in metadata_entities:
            format_entity(node)
    
    lines.append("\n    %% Core Category Relationships")
    # Add relationships with cardinality
    for rel in schema.relationship_types:
        for source in rel.source_labels:
            for target in rel.target_labels:
                rel_key = f"{source}_{rel.type}_{target}"
                if rel_key not in added_relationships:
                    # Determine cardinality based on relationship type
                    cardinality = "||--o{"  # default: one-to-many optional
                    
                    # Special cases for known relationship types
                    if rel.type in ["BELONGS_TO", "CURRENT_VERSION"]:
                        cardinality = "||--||"  # one-to-one
                    elif rel.type in ["HAS_BUREAU", "HAS_USE_CASE"]:
                        cardinality = "||--|{"  # one-to-many required
                    elif rel.type in ["TAGGED_WITH", "HAS_CAPABILITY"]:
                        cardinality = "||--o{"  # one-to-many optional
                    
                    # Add relationship
                    lines.append(f"    {source} {cardinality} {target} : {rel.type}")
                    
                    # Add relationship properties as comments if they exist
                    if rel.properties:
                        lines.append("    %% Properties:")
                        for prop, prop_type in rel.properties.items():
                            lines.append(f"    %% - {prop}: {prop_type}")
                    
                    added_relationships.add(rel_key)
    
    lines.append("```")
    return "\n".join(lines)

def generate_schema_visualization(schema: DatabaseSchema) -> str:
    """Generate a complete schema visualization document"""
    lines = [
        "# Neo4j Database Schema Visualization",
        f"\nGenerated at: {schema.timestamp}\n",
        "## Schema Diagram",
        "\nThis diagram shows the complete database schema including entities, relationships, and their properties.\n",
        "### Legend",
        "- Boxes represent entities (node labels)",
        "- Lines represent relationships between entities",
        "- Relationship cardinality is shown using crow's foot notation:",
        "  - `||` One",
        "  - `o|` Zero or one",
        "  - `}|` One or many",
        "  - `}o` Zero or many",
        "\n### Cardinality Examples",
        "- `||--||` One-to-one relationship",
        "- `||--|{` One-to-many relationship",
        "- `}o--o{` Many-to-many relationship",
        "- `||--o{` One-to-zero-or-many relationship",
        "\n## Database Schema Diagram\n"
    ]
    
    # Add the Mermaid diagram
    lines.append(generate_mermaid_diagram(schema))
    
    # Add statistics and metadata
    lines.extend([
        "\n## Schema Statistics",
        f"\n- Total Node Types: {len(schema.node_labels)}",
        f"- Total Relationship Types: {len(schema.relationship_types)}",
        "\n### Node Types Overview"
    ])
    
    # Add detailed node information
    for node in schema.node_labels:
        lines.extend([
            f"\n#### {node.label}",
            f"- Total Properties: {len(node.properties)}",
            "- Properties:",
            "  ```",
            *[f"  {prop}: {type}" for prop, type in node.properties.items()],
            "  ```",
            f"- Outgoing Relationships: {len(node.relationships_out)}",
            "  ```",
            *[f"  - {rel}" for rel in sorted(node.relationships_out)],
            "  ```",
            f"- Incoming Relationships: {len(node.relationships_in)}",
            "  ```",
            *[f"  - {rel}" for rel in sorted(node.relationships_in)],
            "  ```"
        ])
    
    # Add relationship type details
    lines.extend(["\n### Relationship Types Overview"])
    
    for rel in schema.relationship_types:
        lines.extend([
            f"\n#### {rel.type}",
            "- Properties:",
            "  ```",
            *[f"  {prop}: {type}" for prop, type in rel.properties.items()],
            "  ```",
            "- Source Node Labels:",
            "  ```",
            *[f"  - {label}" for label in sorted(rel.source_labels)],
            "  ```",
            "- Target Node Labels:",
            "  ```",
            *[f"  - {label}" for label in sorted(rel.target_labels)],
            "  ```"
        ])
    
    return "\n".join(lines)

def backup_existing_files(docs_dir: Path) -> None:
    """Backup existing schema files"""
    backup_dir = docs_dir / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        'neo4j_schema_documentation.md',
        'neo4j_schema.json'
    ]

    for file in files_to_backup:
        source = docs_dir / file
        if source.exists():
            logger.info(f"Backed up {file} to {backup_dir}")
            source.rename(backup_dir / file)

def main():
    """Main function to generate schema documentation"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        logger.info("Extracting schema information from database...")
        extractor = SchemaExtractor(neo4j_uri, neo4j_user, neo4j_password)
        schema = extractor.get_schema()
        extractor.close()

        # Create docs directory if it doesn't exist
        docs_dir = Path('docs/neo4j')
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Backup existing files
        logger.info("Backing up existing schema files...")
        backup_existing_files(docs_dir)

        # Generate and save markdown documentation
        logger.info("Generating markdown documentation...")
        markdown_content = generate_markdown(schema)
        with open(docs_dir / 'neo4j_schema_documentation.md', 'w') as f:
            f.write(markdown_content)

        # Generate and save schema visualization
        logger.info("Generating schema visualization...")
        visualization_content = generate_schema_visualization(schema)
        with open(docs_dir / 'neo4j_schema_visualization.md', 'w') as f:
            f.write(visualization_content)

        # Save JSON schema
        logger.info("Saving JSON schema...")
        with open(docs_dir / 'neo4j_schema.json', 'w') as f:
            json.dump(schema.to_dict(), f, indent=2, cls=SchemaEncoder)

        logger.info("Schema documentation generated successfully!")

    except Exception as e:
        logger.error(f"Failed to generate schema documentation: {e}")
        raise

if __name__ == '__main__':
    exit(main()) 