#!/usr/bin/env python3
"""
Agency Use Case Listing Script

Lists all use cases for a given agency abbreviation, showing details like ID, agency, use case name,
purpose/benefits, and outputs.

Usage:
    python list_agency_use_cases.py <agency_abbreviation>

Example:
    python list_agency_use_cases.py TVA

Environment Variables:
    NEO4J_URI: Neo4j database URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tabulate import tabulate

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
        """Close the Neo4j connection"""
        self.driver.close()

    def get_agency_use_cases(self, agency_abbreviation: str) -> List[Dict]:
        """
        Get all use cases for the specified agency abbreviation.
        
        Args:
            agency_abbreviation (str): The agency's abbreviation (e.g., 'TVA')
            
        Returns:
            List[Dict]: List of use cases with their details
        """
        # First, let's verify if the agency exists and print debug info
        debug_query = """
        MATCH (a:Agency)
        WHERE toUpper(a.abbreviation) = toUpper($abbreviation)
        RETURN a.name as name, a.abbreviation as abbreviation
        """
        
        with self.driver.session() as session:
            debug_result = session.run(debug_query, abbreviation=agency_abbreviation)
            agency = debug_result.single()
            if agency:
                print(f"Found agency: {agency['name']} ({agency['abbreviation']})")
            else:
                print(f"No agency found with abbreviation: {agency_abbreviation}")
        
        # Main query using the correct BELONGS_TO relationship
        query = """
        MATCH (u:UseCase)-[:BELONGS_TO]->(a:Agency)
        WHERE toUpper(a.abbreviation) = toUpper($abbreviation)
        OPTIONAL MATCH (u)-[:HAS_PURPOSE]->(p:PurposeBenefit)
        OPTIONAL MATCH (u)-[:PRODUCES]->(o:Output)
        RETURN 
            u.id as id,
            a.name as agency_name,
            a.abbreviation as agency_abbreviation,
            u.name as use_case_name,
            COALESCE(p.description, u.purpose_benefits) as purpose_benefits,
            COALESCE(o.description, u.outputs) as outputs
        ORDER BY u.name
        """
        
        with self.driver.session() as session:
            result = session.run(query, abbreviation=agency_abbreviation)
            return [record.data() for record in result]

def format_text(text: str, max_length: int = 50) -> str:
    """Format text for display by wrapping at word boundaries"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    wrapped = text[:max_length]
    last_space = wrapped.rfind(" ")
    if last_space > 0:
        return text[:last_space] + "..."
    return wrapped + "..."

def display_use_cases(use_cases: List[Dict]) -> None:
    """Display use cases in a formatted table"""
    if not use_cases:
        print(f"No use cases found for the specified agency.")
        return

    # Format the data for tabulate
    headers = ["ID", "Agency", "Use Case Name", "Purpose/Benefits", "Outputs"]
    rows = []
    
    for case in use_cases:
        rows.append([
            case.get("id", "N/A"),
            f"{case.get('agency_name', 'N/A')} ({case.get('agency_abbreviation', 'N/A')})",
            format_text(case.get("use_case_name", "N/A")),
            format_text(case.get("purpose_benefits", "N/A")),
            format_text(case.get("outputs", "N/A"))
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nTotal use cases found: {len(use_cases)}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <agency_abbreviation>")
        print(f"Example: python {sys.argv[0]} TVA")
        sys.exit(1)

    agency_abbreviation = sys.argv[1]
    
    try:
        neo4j = Neo4jConnection()
        use_cases = neo4j.get_agency_use_cases(agency_abbreviation)
        display_use_cases(use_cases)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        neo4j.close()

if __name__ == "__main__":
    main() 