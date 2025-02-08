#!/usr/bin/env python3
"""Fix duplicate agency nodes in Neo4j"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_duplicate_agencies():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set")
        return
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # 1. Find duplicate agencies
            logger.info("\nFinding duplicate agencies:")
            result = session.run("""
                MATCH (a:Agency)
                WITH a.abbreviation as abbr, collect(a) as agencies
                WHERE size(agencies) > 1
                RETURN abbr, [agency in agencies | agency.name] as names
            """)
            duplicates = [(record["abbr"], record["names"]) for record in result]
            
            if not duplicates:
                logger.info("No duplicate agencies found!")
                return
            
            for abbr, names in duplicates:
                logger.info(f"\nFound duplicates for {abbr}:")
                for name in names:
                    logger.info(f"  - {name}")
                
                # 2. Merge duplicates
                logger.info(f"\nMerging duplicates for {abbr}...")
                
                # Use proper case name as canonical
                canonical_name = max(names, key=lambda x: sum(1 for c in x if c.isupper()))
                
                result = session.run("""
                    // Find all agency nodes with this abbreviation
                    MATCH (a:Agency)
                    WHERE a.abbreviation = $abbr
                    
                    // Get the canonical agency node
                    WITH collect(a) as agencies,
                         head([a in collect(a) WHERE a.name = $canonical_name]) as canonical
                    
                    // For each other agency node
                    UNWIND [a in agencies WHERE a <> canonical] as other
                    
                    // Move all relationships to canonical node
                    CALL apoc.refactor.mergeNodes([canonical, other], {
                        properties: "combine",
                        mergeRels: true
                    })
                    YIELD node
                    
                    RETURN node.name as name, node.abbreviation as abbr
                """, abbr=abbr, canonical_name=canonical_name)
                
                try:
                    record = result.single()
                    if record:
                        logger.info(f"Successfully merged into: {record['name']} ({record['abbr']})")
                    else:
                        logger.warning(f"No merge result for {abbr}")
                except Exception as e:
                    logger.error(f"Error processing merge result: {str(e)}")
        
        driver.close()
        logger.info("\nDuplicate agency cleanup completed!")
        
    except Exception as e:
        logger.error(f"Error fixing duplicate agencies: {str(e)}")

if __name__ == "__main__":
    fix_duplicate_agencies() 