#!/usr/bin/env python3
"""
List Agency Abbreviations Script

Lists all agencies and their abbreviations in the Neo4j database.

Usage:
    python list_agency_abbreviations.py

Environment Variables:
    NEO4J_URI: Neo4j database URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import os
from typing import List, Dict
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

    def get_all_agencies(self) -> List[Dict]:
        """
        Get all agencies and their abbreviations.
        
        Returns:
            List[Dict]: List of agencies with their names and abbreviations
        """
        query = """
        MATCH (a:Agency)
        WITH a, size((a)<-[:BELONGS_TO]-()) as use_case_count
        RETURN 
            a.name as name,
            a.abbreviation as abbreviation,
            use_case_count
        ORDER BY a.name
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

def display_agencies(agencies: List[Dict]) -> None:
    """Display agencies in a formatted table"""
    if not agencies:
        print("No agencies found in the database.")
        return

    # Format the data for tabulate
    headers = ["Agency Name", "Abbreviation", "Number of Use Cases"]
    rows = [[
        agency.get("name", "N/A"),
        agency.get("abbreviation", "N/A"),
        agency.get("use_case_count", 0)
    ] for agency in agencies]
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nTotal agencies found: {len(agencies)}")

def main():
    try:
        neo4j = Neo4jConnection()
        agencies = neo4j.get_all_agencies()
        display_agencies(agencies)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        neo4j.close()

if __name__ == "__main__":
    main() 