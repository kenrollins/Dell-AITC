#!/usr/bin/env python
"""
Update Neo4j database with latest AI technology categories
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from setup.initialize_database import DatabaseInitializer

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    
    # Initialize database connection
    db = DatabaseInitializer(uri, user, password)
    
    try:
        # Verify connection
        if not db.verify_connection():
            raise ConnectionError("Could not connect to Neo4j database")
            
        # Get paths - fix to use parent.parent.parent to get to project root
        base_path = Path(__file__).parent.parent.parent
        categories_file = base_path / 'data' / 'input' / 'AI-Technology-Categories-v1.5.csv'
        
        print(f"Updating categories from: {categories_file}")
        print(f"File exists: {categories_file.exists()}")
        
        # Load new categories
        if db.load_categories(categories_file):
            print("Successfully updated categories in Neo4j")
        else:
            print("Failed to update categories")
            
    except Exception as e:
        print(f"Error updating categories: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 