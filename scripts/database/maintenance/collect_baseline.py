#!/usr/bin/env python
"""
Baseline Metrics Collector for Schema Migration
Collects and stores comprehensive baseline metrics before schema updates.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from schema_monitor import SchemaMonitor
from schema_validator import SchemaValidator
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineCollector:
    """Collects and stores baseline metrics for schema migration."""
    
    def __init__(self, 
                 schema_path: Path,
                 metrics_dir: Path,
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None):
        """Initialize the baseline collector.
        
        Args:
            schema_path: Path to current schema JSON
            metrics_dir: Directory to store baseline metrics
            neo4j_uri: Optional Neo4j database URI
            neo4j_user: Optional database username
            neo4j_password: Optional database password
        """
        self.schema_path = schema_path
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.has_db_connection = all([neo4j_uri, neo4j_user, neo4j_password])
        if self.has_db_connection:
            self.monitor = SchemaMonitor(
                uri=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password,
                metrics_dir=metrics_dir
            )
        self.validator = SchemaValidator(schema_path)
        
    def collect_schema_validation(self) -> dict:
        """Run schema validation and collect results."""
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'valid': self.validator.validate_all(),
            'errors': self.validator.get_validation_errors(),
            'schema_file': str(self.schema_path)
        }
        
        # Save validation results
        validation_path = self.metrics_dir / 'schema_validation.json'
        with open(validation_path, 'w') as f:
            json.dump(validation_result, f, indent=2)
            
        return validation_result
        
    def analyze_schema_structure(self) -> dict:
        """Analyze the schema structure without requiring database access."""
        schema = self.validator.load_schema()
        if not schema:
            return {}
            
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'node_types': list(schema.get('nodes', {}).keys()),
            'relationship_types': list(schema.get('relationships', {}).keys()),
            'property_counts': {},
            'constraints': [],
            'indexes': []
        }
        
        # Analyze properties per node type
        for node_type, node_def in schema.get('nodes', {}).items():
            analysis['property_counts'][node_type] = len(node_def.get('properties', {}))
            
        # Extract constraints and indexes
        for node_type, node_def in schema.get('nodes', {}).items():
            props = node_def.get('properties', {})
            for prop_name, prop_def in props.items():
                if prop_def.get('unique', False):
                    analysis['constraints'].append(f"{node_type}.{prop_name} UNIQUE")
                if prop_def.get('indexed', False):
                    analysis['indexes'].append(f"{node_type}.{prop_name}")
                    
        return analysis
        
    def collect_baseline_metrics(self):
        """Collect all baseline metrics."""
        logger.info("Collecting baseline metrics...")
        
        # Always collect schema validation
        validation = self.collect_schema_validation()
        
        # Always analyze schema structure
        analysis = self.analyze_schema_structure()
        
        # Save schema analysis
        analysis_path = self.metrics_dir / 'schema_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Collect database metrics if connection available
        metrics = None
        if self.has_db_connection:
            metrics = self.monitor.collect_metrics()
            self.monitor.save_metrics(metrics)
            
            # Generate and save report
            report = self.monitor.generate_report(metrics)
            report_path = self.metrics_dir / 'baseline_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
        
        logger.info(f"Baseline collection completed and saved to {self.metrics_dir}")
        return metrics, validation, analysis
        
    def generate_summary(self, metrics, validation, analysis) -> str:
        """Generate a summary of baseline metrics and validation."""
        summary = []
        summary.append("Schema Analysis Summary")
        summary.append("=====================")
        summary.append(f"\nCollected at: {datetime.now().isoformat()}")
        
        # Validation status
        summary.append("\nSchema Validation:")
        summary.append(f"Status: {'PASS' if validation['valid'] else 'FAIL'}")
        if validation['errors']:
            summary.append("Errors:")
            for error in validation['errors']:
                summary.append(f"  - {error}")
                
        # Schema structure
        summary.append("\nSchema Structure:")
        summary.append(f"Node Types: {', '.join(analysis['node_types'])}")
        summary.append(f"Relationship Types: {', '.join(analysis['relationship_types'])}")
        
        summary.append("\nProperty Counts:")
        for node_type, count in analysis['property_counts'].items():
            summary.append(f"  {node_type}: {count} properties")
            
        summary.append("\nConstraints:")
        for constraint in analysis['constraints']:
            summary.append(f"  - {constraint}")
            
        summary.append("\nIndexes:")
        for index in analysis['indexes']:
            summary.append(f"  - {index}")
            
        # Database metrics if available
        if metrics:
            summary.append("\nDatabase Metrics:")
            summary.append("(Database currently empty - no metrics to report)")
            
        return "\n".join(summary)

def main():
    """CLI entry point for baseline collection."""
    import argparse
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Collect schema baseline metrics')
    parser.add_argument('--schema-path', type=Path, required=True,
                       help='Path to current schema JSON')
    parser.add_argument('--metrics-dir', type=Path,
                       default=Path('metrics/schema/v2.0-baseline'),
                       help='Directory to store baseline metrics')
    args = parser.parse_args()
    
    collector = BaselineCollector(
        schema_path=args.schema_path,
        metrics_dir=args.metrics_dir
    )
    
    try:
        metrics, validation, analysis = collector.collect_baseline_metrics()
        summary = collector.generate_summary(metrics, validation, analysis)
        print("\nBaseline Collection Summary:")
        print(summary)
    except Exception as e:
        logger.error(f"Failed to collect baseline metrics: {e}")
        
if __name__ == '__main__':
    main() 