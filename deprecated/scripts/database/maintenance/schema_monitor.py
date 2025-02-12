#!/usr/bin/env python
"""
Schema Monitor for Dell-AITC
Tracks schema changes and performance metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from neo4j import GraphDatabase
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SchemaMetrics:
    """Container for schema metrics."""
    node_counts: Dict[str, int]
    relationship_counts: Dict[str, int]
    property_usage: Dict[str, Dict[str, int]]
    query_performance: Dict[str, float]
    timestamp: datetime

class SchemaMonitor:
    """Monitors Neo4j schema changes and performance."""
    
    def __init__(self, uri: str, username: str, password: str, 
                 metrics_dir: Optional[Path] = None):
        """Initialize the schema monitor.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
            metrics_dir: Directory to store metrics data
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.metrics_dir = metrics_dir or Path.cwd() / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_node_metrics(self) -> Dict[str, int]:
        """Collect node count metrics for each label."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n:`$label`)
                    RETURN count(n) as count
                }
                RETURN label, count
            """)
            return {record["label"]: record["count"] for record in result}
            
    def collect_relationship_metrics(self) -> Dict[str, int]:
        """Collect relationship count metrics for each type."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL {
                    WITH relationshipType
                    MATCH ()-[r:`$relationshipType`]->()
                    RETURN count(r) as count
                }
                RETURN relationshipType, count
            """)
            return {record["relationshipType"]: record["count"] for record in result}
            
    def collect_property_usage(self) -> Dict[str, Dict[str, int]]:
        """Collect property usage statistics."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.propertyKeys() YIELD propertyKey
                CALL {
                    WITH propertyKey
                    MATCH (n)
                    WHERE exists(n[propertyKey])
                    RETURN labels(n) as labels, count(n) as count
                }
                RETURN propertyKey, labels, count
            """)
            
            usage = {}
            for record in result:
                prop = record["propertyKey"]
                labels = ":".join(record["labels"])
                count = record["count"]
                
                if prop not in usage:
                    usage[prop] = {}
                usage[prop][labels] = count
                
            return usage
            
    def measure_query_performance(self) -> Dict[str, float]:
        """Measure performance of common queries."""
        test_queries = {
            "category_lookup": "MATCH (c:AICategory) RETURN c LIMIT 1",
            "keyword_search": "MATCH (k:Keyword) RETURN k LIMIT 100",
            "relationship_traversal": """
                MATCH (c:AICategory)-[:HAS_KEYWORD]->(k:Keyword)
                RETURN c, k LIMIT 100
            """
        }
        
        performance = {}
        with self.driver.session() as session:
            for name, query in test_queries.items():
                start = datetime.now()
                session.run(query)
                duration = (datetime.now() - start).total_seconds()
                performance[name] = duration
                
        return performance
        
    def collect_metrics(self) -> SchemaMetrics:
        """Collect all schema metrics."""
        return SchemaMetrics(
            node_counts=self.collect_node_metrics(),
            relationship_counts=self.collect_relationship_metrics(),
            property_usage=self.collect_property_usage(),
            query_performance=self.measure_query_performance(),
            timestamp=datetime.now()
        )
        
    def save_metrics(self, metrics: SchemaMetrics):
        """Save metrics to CSV files."""
        timestamp = metrics.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save node counts
        pd.DataFrame(list(metrics.node_counts.items()),
                    columns=['label', 'count']).to_csv(
            self.metrics_dir / f"node_counts_{timestamp}.csv",
            index=False
        )
        
        # Save relationship counts
        pd.DataFrame(list(metrics.relationship_counts.items()),
                    columns=['type', 'count']).to_csv(
            self.metrics_dir / f"relationship_counts_{timestamp}.csv",
            index=False
        )
        
        # Save property usage
        property_rows = []
        for prop, usages in metrics.property_usage.items():
            for labels, count in usages.items():
                property_rows.append({
                    'property': prop,
                    'labels': labels,
                    'count': count
                })
        pd.DataFrame(property_rows).to_csv(
            self.metrics_dir / f"property_usage_{timestamp}.csv",
            index=False
        )
        
        # Save query performance
        pd.DataFrame(list(metrics.query_performance.items()),
                    columns=['query', 'duration']).to_csv(
            self.metrics_dir / f"query_performance_{timestamp}.csv",
            index=False
        )
        
    def generate_report(self, metrics: SchemaMetrics) -> str:
        """Generate a human-readable report of the metrics."""
        report = []
        report.append("Schema Monitoring Report")
        report.append(f"Generated at: {metrics.timestamp}")
        report.append("\nNode Counts:")
        for label, count in metrics.node_counts.items():
            report.append(f"  {label}: {count}")
            
        report.append("\nRelationship Counts:")
        for rel_type, count in metrics.relationship_counts.items():
            report.append(f"  {rel_type}: {count}")
            
        report.append("\nQuery Performance:")
        for query, duration in metrics.query_performance.items():
            report.append(f"  {query}: {duration:.3f}s")
            
        return "\n".join(report)

def main():
    """CLI entry point for schema monitoring."""
    import argparse
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Monitor Neo4j schema')
    parser.add_argument('--metrics-dir', type=Path,
                       help='Directory to store metrics',
                       default='./metrics')
    args = parser.parse_args()
    
    monitor = SchemaMonitor(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD'),
        metrics_dir=args.metrics_dir
    )
    
    try:
        metrics = monitor.collect_metrics()
        monitor.save_metrics(metrics)
        print(monitor.generate_report(metrics))
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        
if __name__ == '__main__':
    main() 