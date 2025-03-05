#!/usr/bin/env python3
"""
Setup script for keyword types and relevance scores.

This script:
1. Updates existing keywords to have proper types (technical_keywords, capabilities, business_language)
2. Sets default relevance scores for keywords
3. Creates the USES_TECHNOLOGY relationship type if missing

Usage:
    python setup_keywords.py [--dry-run]
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_neo4j_connection():
    """Initialize Neo4j connection"""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, user, password]):
        raise ValueError("Missing required Neo4j environment variables")
        
    return GraphDatabase.driver(uri, auth=(user, password))

def update_keyword_types(session, dry_run: bool = True):
    """Update keyword types based on patterns and context"""
    # First, get all keywords
    result = session.run("MATCH (k:Keyword) RETURN k.name as name")
    keywords = [record["name"] for record in result]
    
    # Define patterns for each type
    technical_patterns = [
        "algorithm", "api", "automation", "cloud", "data", "deep learning",
        "machine learning", "ml", "model", "neural", "nlp", "processing",
        "recognition", "training", "transformer", "vision"
    ]
    
    capability_patterns = [
        "analysis", "detection", "classification", "prediction", "generation",
        "optimization", "recommendation", "scheduling", "scoring", "search",
        "segmentation", "synthesis", "tracking"
    ]
    
    business_patterns = [
        "benefits", "compliance", "cost", "efficiency", "governance",
        "improvement", "management", "monitoring", "performance", "planning",
        "quality", "risk", "strategy", "support"
    ]
    
    updates = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Determine type based on patterns
        if any(pattern in keyword_lower for pattern in technical_patterns):
            type = "technical_keywords"
        elif any(pattern in keyword_lower for pattern in capability_patterns):
            type = "capabilities"
        elif any(pattern in keyword_lower for pattern in business_patterns):
            type = "business_language"
        else:
            # Default to technical if no clear match
            type = "technical_keywords"
            
        # Calculate a basic relevance score (can be refined later)
        # Longer, more specific terms get higher scores
        base_score = min(0.8, max(0.4, len(keyword.split()) * 0.2))
        
        updates.append((keyword, type, base_score))
    
    if dry_run:
        logging.info("\nWould update keyword types and scores:")
        type_counts = {"technical_keywords": 0, "capabilities": 0, "business_language": 0}
        for keyword, type, score in updates:
            type_counts[type] += 1
            logging.info(f"  - {keyword}: {type} (score: {score:.2f})")
        
        logging.info("\nSummary:")
        for type, count in type_counts.items():
            logging.info(f"  {type}: {count} keywords")
    else:
        # Perform the updates
        for keyword, type, score in updates:
            session.run("""
                MATCH (k:Keyword {name: $name})
                SET k.type = $type,
                    k.relevance_score = $score
                """,
                {"name": keyword, "type": type, "score": score}
            )
        logging.info("✓ Updated keyword types and relevance scores")

def ensure_relationship_types(session, dry_run: bool = True):
    """Ensure required relationship types exist"""
    if dry_run:
        logging.info("\nWould create relationship type: USES_TECHNOLOGY")
    else:
        # Create USES_TECHNOLOGY relationship type by creating and deleting a test relationship
        session.run("""
            CREATE (a:_TestNode)
            CREATE (b:_TestNode)
            CREATE (a)-[r:USES_TECHNOLOGY]->(b)
            DELETE a, r, b
        """)
        logging.info("✓ Created USES_TECHNOLOGY relationship type")

def main():
    parser = argparse.ArgumentParser(description="Setup keyword types and relevance scores")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    load_dotenv()
    
    try:
        driver = setup_neo4j_connection()
        with driver.session() as session:
            update_keyword_types(session, args.dry_run)
            ensure_relationship_types(session, args.dry_run)
            
        if args.dry_run:
            logging.info("\nDRY RUN COMPLETED - No changes were made")
        else:
            logging.info("\nSetup completed successfully")
            
    except Exception as e:
        logging.error(f"Error during setup: {str(e)}")
        return 1
    finally:
        driver.close()
    
    return 0

if __name__ == "__main__":
    exit(main()) 