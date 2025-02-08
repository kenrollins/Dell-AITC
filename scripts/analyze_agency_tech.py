#!/usr/bin/env python3
"""
Agency AI Technology Analysis Script

This script analyzes an agency's or bureau's use cases and their AI technology patterns,
providing insights into which AI categories are most relevant for their needs.

Usage:
    python analyze_agency_tech.py (-a | --agency) <abbreviation>
    python analyze_agency_tech.py (-b | --bureau) <name>

Examples:
    python analyze_agency_tech.py --agency TVA
    python analyze_agency_tech.py --bureau "Bureau of Land Management"

Environment Variables:
    NEO4J_URI: Neo4j database URI
    NEO4J_USER: Neo4j username
    NEO4J_PASSWORD: Neo4j password
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )

    def close(self):
        self.driver.close()

    def get_agency_info(self, abbreviation: str) -> Optional[dict]:
        """Get basic agency information"""
        query = """
        MATCH (a:Agency)
        WHERE toUpper(a.abbreviation) = toUpper($abbreviation)
        RETURN a.name as name, a.abbreviation as abbreviation
        """
        with self.driver.session() as session:
            result = session.run(query, abbreviation=abbreviation)
            record = result.single()
            return record.data() if record else None

    def get_bureau_info(self, name: str) -> List[dict]:
        """Get bureau information including parent agencies"""
        query = """
        MATCH (b:Bureau)<-[:HAS_BUREAU]-(a:Agency)
        WHERE b.name CONTAINS $name
        RETURN b.name as bureau_name, 
               collect({name: a.name, abbreviation: a.abbreviation}) as agencies
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            return [record.data() for record in result]

    def get_agency_bureaus(self, abbreviation: str) -> List[str]:
        """Get list of bureaus for the agency"""
        query = """
        MATCH (a:Agency)-[:HAS_BUREAU]->(b:Bureau)
        WHERE toUpper(a.abbreviation) = toUpper($abbreviation)
        RETURN b.name as bureau_name
        ORDER BY b.name
        """
        with self.driver.session() as session:
            result = session.run(query, abbreviation=abbreviation)
            return [record["bureau_name"] for record in result]

    def get_use_cases(self, entity_type: str, entity_value: str) -> List[dict]:
        """Get all use cases for the agency or bureau with their details"""
        if entity_type == "agency":
            match_clause = "MATCH (a:Agency)-[:HAS_USE_CASE]->(u:UseCase) WHERE toUpper(a.abbreviation) = toUpper($value)"
        else:  # bureau
            match_clause = "MATCH (b:Bureau)<-[:HAS_BUREAU]-(a:Agency)-[:HAS_USE_CASE]->(u:UseCase) WHERE b.name CONTAINS $value"
        
        query = f"""
        {match_clause}
        OPTIONAL MATCH (u)-[:HAS_PURPOSE]->(p:PurposeBenefit)
        OPTIONAL MATCH (u)-[:PRODUCES]->(o:Output)
        OPTIONAL MATCH (u)-[:USES_SYSTEM]->(s:System)
        WITH u, 
             collect(DISTINCT COALESCE(p.description, '')) as purposes,
             collect(DISTINCT COALESCE(o.description, '')) as outputs,
             collect(DISTINCT COALESCE(s.name, '')) as systems
        RETURN 
            COALESCE(u.name, '') as name,
            COALESCE(u.topic_area, '') as topic_area,
            COALESCE(u.dev_stage, '') as dev_stage,
            purposes,
            outputs,
            systems
        ORDER BY name
        """
        with self.driver.session() as session:
            result = session.run(query, value=entity_value)
            return [record.data() for record in result]

    def get_ai_categories(self, entity_type: str, entity_value: str) -> List[dict]:
        """Get AI categories implemented by use cases with relationship details"""
        if entity_type == "agency":
            match_clause = """
            MATCH (a:Agency)-[:HAS_USE_CASE]->(u:UseCase)-[:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
            WHERE toUpper(a.abbreviation) = toUpper($value)
            """
        else:  # bureau
            match_clause = """
            MATCH (b:Bureau)<-[:HAS_BUREAU]-(a:Agency)-[:HAS_USE_CASE]->(u:UseCase)
                  -[:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
            WHERE b.name CONTAINS $value
            """
        
        query = f"""
        {match_clause}
        WITH c, e, u
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(z:Zone)
        WITH c, e, u, z
        RETURN 
            c.name as category_name,
            c.maturity_level as maturity_level,
            z.name as zone_name,
            e.relationship_type as relationship_type,
            e.confidence as confidence,
            count(DISTINCT u) as use_case_count,
            collect(DISTINCT u.name) as use_case_names
        ORDER BY use_case_count DESC
        """
        
        print(f"Running query for {entity_type}: {entity_value}")  # Debug print
        with self.driver.session() as session:
            result = session.run(query, value=entity_value)
            categories = [record.data() for record in result]
            print(f"Found {len(categories)} categories")  # Debug print
            if len(categories) == 0:
                print("No categories found. Checking agency existence...")  # Debug print
                # Check if agency exists
                check_query = """
                MATCH (a:Agency)
                WHERE toUpper(a.abbreviation) = toUpper($value)
                RETURN a.name as name, a.abbreviation as abbreviation
                """
                check_result = session.run(check_query, value=entity_value)
                agency = check_result.single()
                if agency:
                    print(f"Agency exists: {agency['name']} ({agency['abbreviation']})")
                else:
                    print("Agency not found")
            return categories

    def get_topic_categories(self, entity_type: str, entity_value: str) -> List[dict]:
        """Get AI categories by topic area"""
        if entity_type == "agency":
            match_clause = """
            MATCH (a:Agency)-[:HAS_USE_CASE]->(u:UseCase)-[:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
            WHERE toUpper(a.abbreviation) = toUpper($value)
            """
        else:  # bureau
            match_clause = """
            MATCH (b:Bureau)<-[:HAS_BUREAU]-(a:Agency)-[:HAS_USE_CASE]->(u:UseCase)
                  -[:HAS_EVALUATION]->(e:CategoryEvaluation)-[:EVALUATES]->(c:AICategory)
            WHERE b.name CONTAINS $value
            """
        
        query = f"""
        {match_clause}
        WITH COALESCE(u.topic_area, 'Unspecified') as topic, 
             c.name as category_name, 
             e.relationship_type as rel_type,
             count(*) as category_count
        ORDER BY topic, category_count DESC
        RETURN topic, collect({{
            category: category_name, 
            count: category_count, 
            rel_type: rel_type
        }}) as categories
        """
        with self.driver.session() as session:
            result = session.run(query, value=entity_value)
            return [record.data() for record in result]

def generate_category_distribution_chart(categories: List[dict], output_dir: str):
    """Generate a bar chart showing AI category distribution"""
    if not categories:
        # Create an empty chart with a message
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No AI categories found', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('AI Technology Category Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_distribution.png")
        plt.close()
        return

    # Convert to DataFrame and ensure column names match
    df = pd.DataFrame(categories)
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame head:", df.head())
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='use_case_count', y='category_name')
    plt.title('AI Technology Category Distribution')
    plt.xlabel('Number of Use Cases')
    plt.ylabel('AI Category')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_distribution.png")
    plt.close()

def generate_topic_heatmap(topic_categories: List[dict], output_dir: str):
    """Generate a heatmap showing AI categories by topic area"""
    if not topic_categories:
        # Create an empty chart with a message
        plt.figure(figsize=(15, 8))
        plt.text(0.5, 0.5, 'No topic categories found', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('AI Categories by Topic Area')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_heatmap.png")
        plt.close()
        return

    print("Topic categories data:", topic_categories[:2])
    
    # Transform data for heatmap
    topic_cat_matrix = defaultdict(lambda: defaultdict(int))
    
    for topic_data in topic_categories:
        topic = topic_data['topic'] or 'Unspecified'
        print(f"Processing topic: {topic}")
        print(f"Categories data: {topic_data['categories'][:2]}")
        for cat_data in topic_data['categories']:
            category = cat_data['category']
            count = cat_data['count']
            topic_cat_matrix[topic][category] = count
    
    # If no data was processed, create empty chart
    if not topic_cat_matrix:
        plt.figure(figsize=(15, 8))
        plt.text(0.5, 0.5, 'No data available for heatmap', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('AI Categories by Topic Area')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_heatmap.png")
        plt.close()
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(topic_cat_matrix).fillna(0)
    print("Heatmap DataFrame shape:", df.shape)
    print("Heatmap DataFrame columns:", df.columns.tolist())
    
    # Generate heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(df, annot=True, fmt='g', cmap='YlOrRd')
    plt.title('AI Categories by Topic Area')
    plt.xlabel('Topic Area')
    plt.ylabel('AI Category')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_heatmap.png")
    plt.close()

def analyze_entity(entity_type: str, entity_value: str):
    """Main analysis function for both agencies and bureaus"""
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"data/output/{entity_type}_analysis/{entity_value}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Neo4j connection
    neo4j = Neo4jConnection()
    
    try:
        # Get entity info
        if entity_type == "agency":
            entity_info = neo4j.get_agency_info(entity_value)
            if not entity_info:
                print(f"No agency found with abbreviation {entity_value}")
                return
            
            # Get agency data
            bureaus = neo4j.get_agency_bureaus(entity_value)
            use_cases = neo4j.get_use_cases("agency", entity_value)
            ai_categories = neo4j.get_ai_categories("agency", entity_value)
            topic_categories = neo4j.get_topic_categories("agency", entity_value)
            
            # Generate report header
            report = []
            report.append(f"AI Technology Analysis for {entity_info['name']} ({entity_info['abbreviation']})")
            report.append("=" * 80)
            report.append("")
            
        else:  # bureau
            bureau_info = neo4j.get_bureau_info(entity_value)
            if not bureau_info:
                print(f"No bureau found containing '{entity_value}'")
                return
            
            # Get bureau data
            use_cases = neo4j.get_use_cases("bureau", entity_value)
            ai_categories = neo4j.get_ai_categories("bureau", entity_value)
            topic_categories = neo4j.get_topic_categories("bureau", entity_value)
            
            # Generate report header
            report = []
            for bureau in bureau_info:
                report.append(f"AI Technology Analysis for Bureau: {bureau['bureau_name']}")
                report.append("Parent Agencies:")
                for agency in bureau['agencies']:
                    report.append(f"- {agency['name']} ({agency['abbreviation']})")
                report.append("=" * 80)
                report.append("")
        
        # Basic statistics
        report.append("Basic Statistics")
        report.append("-" * 20)
        if entity_type == "agency":
            report.append(f"Total Bureaus: {len(bureaus)}")
        report.append(f"Total Use Cases: {len(use_cases)}")
        report.append(f"AI Categories Implemented: {len(ai_categories)}")
        report.append("")
        
        # Bureau listing for agencies
        if entity_type == "agency" and bureaus:
            report.append("Bureaus")
            report.append("-" * 10)
            for bureau in bureaus:
                report.append(f"- {bureau}")
            report.append("")
        
        # AI Category Analysis
        report.append("AI Technology Category Analysis")
        report.append("-" * 30)
        if ai_categories:
            for category in ai_categories:
                report.append(f"\nCategory: {category['category_name']}")
                report.append(f"Zone: {category['zone_name'] or 'N/A'}")
                report.append(f"Maturity Level: {category['maturity_level']}")
                report.append(f"Use Cases: {category['use_case_count']}")
                report.append(f"Relationship Types: {category['relationship_type']}")
                report.append("Use Case Examples:")
                for uc in category['use_case_names'][:3]:  # Show first 3 examples
                    report.append(f"  - {uc}")
                report.append("")
        else:
            report.append("\nNo AI categories found for this entity.")
            report.append("")
        
        # Topic Area Analysis
        report.append("\nTopic Area Analysis")
        report.append("-" * 20)
        if topic_categories:
            for topic_data in topic_categories:
                report.append(f"\nTopic: {topic_data['topic'] or 'Unspecified'}")
                for cat_data in topic_data['categories'][:5]:  # Top 5 categories per topic
                    report.append(f"  - {cat_data['category']}: {cat_data['count']} use cases")
        else:
            report.append("\nNo topic categories found for this entity.")
            report.append("")
        
        # Save report
        report_path = f"{output_dir}/analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        # Generate visualizations
        generate_category_distribution_chart(ai_categories, output_dir)
        generate_topic_heatmap(topic_categories, output_dir)
        
        print(f"\nAnalysis completed! Results saved to: {output_dir}")
        print(f"- Report: {report_path}")
        print(f"- Visualizations: {output_dir}/category_distribution.png")
        print(f"                  {output_dir}/topic_heatmap.png")
        
    finally:
        neo4j.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze AI technology patterns for federal agencies and bureaus.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--agency', help='Agency abbreviation (e.g., TVA, DOD)')
    group.add_argument('-b', '--bureau', help='Bureau name (can be partial)')
    
    args = parser.parse_args()
    
    if args.agency:
        analyze_entity("agency", args.agency)
    else:
        analyze_entity("bureau", args.bureau)

if __name__ == "__main__":
    main() 