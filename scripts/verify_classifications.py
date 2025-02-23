#!/usr/bin/env python3
"""
Script to verify classifications in Neo4j database.

Usage:
    python verify_classifications.py <use_case_id>

This script verifies the classifications stored in Neo4j for a given use case,
including both regular classifications and no-match analysis if present.
"""

import os
import logging
import asyncio
from neo4j import GraphDatabase
from dotenv import load_dotenv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from neo4j.exceptions import Neo4jError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassificationVerifier:
    """Verifies classifications in Neo4j database"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "kuxFc8HN")
        
        # Initialize driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        """Close the database connection"""
        self.driver.close()
        
    def verify_use_case(self, use_case_id: str) -> Dict[str, Any]:
        """Verify classifications for a use case.
        
        Args:
            use_case_id: ID of the use case to verify
            
        Returns:
            Dictionary containing verification results
        """
        try:
            with self.driver.session() as session:
                # First verify the use case exists
                result = session.run("""
                    MATCH (u:UseCase {id: $id})
                    RETURN u.name as name, u.purpose_benefits as purpose_benefits
                """, id=use_case_id)
                
                use_case = result.single()
                if not use_case:
                    logger.error(f"Use case with ID {use_case_id} not found")
                    return
                    
                logger.info(f"Verifying classifications for use case: {use_case['name']}")
                
                # Get classifications
                result = session.run("""
                    MATCH (u:UseCase {id: $id})-[r:CLASSIFIED_AS]->(c:AICategory)
                    RETURN 
                        r.match_type as match_type,
                        c.name as category,
                        r.confidence as confidence,
                        r.reasoning as reasoning,
                        r.classified_at as classified_at,
                        r.classified_by as classified_by
                    ORDER BY r.confidence DESC
                """, id=use_case_id)
                
                records = list(result)
                if not records:
                    logger.info("No classifications found for this use case")
                    return
                    
                # Print classifications
                for record in records:
                    print(f"\nMatch Type: {record['match_type']}")
                    print(f"Category: {record['category']}")
                    print(f"Confidence: {record['confidence']:.2f}")
                    print(f"Reasoning: {record['reasoning']}")
                    print(f"Classified At: {record['classified_at']}")
                    print(f"Classified By: {record['classified_by']}")
                    
                # Get no-match analysis if it exists
                result = session.run("""
                    MATCH (u:UseCase {id: $id})-[r:HAS_ANALYSIS]->(n:NoMatchAnalysis)
                    RETURN 
                        n.reason as reason,
                        n.llm_analysis as llm_analysis,
                        n.suggested_keywords as suggested_keywords,
                        n.improvement_suggestions as improvement_suggestions,
                        n.created_at as timestamp
                """, id=use_case_id)
                
                no_match = result.single()
                if no_match:
                    print("\nNo-Match Analysis:")
                    print(f"Reason: {no_match['reason']}")
                    print(f"LLM Analysis: {no_match['llm_analysis']}")
                    print(f"Suggested Keywords: {no_match['suggested_keywords']}")
                    print(f"Improvement Suggestions: {no_match['improvement_suggestions']}")
                    print(f"Timestamp: {no_match['timestamp']}")
                    
            return {
                "use_case": {
                    "name": use_case['name'],
                    "purpose_benefits": use_case['purpose_benefits']
                },
                "classifications": [
                    {
                        "match_type": record['match_type'],
                        "category": record['category'],
                        "confidence": record['confidence'],
                        "reasoning": record['reasoning'],
                        "classified_at": record['classified_at'],
                        "classified_by": record['classified_by']
                    } for record in records
                ],
                "no_match_analysis": no_match
            }
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return None

def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python verify_classifications.py <use_case_id>")
        sys.exit(1)
        
    use_case_id = sys.argv[1]
    verifier = ClassificationVerifier()
    
    try:
        results = verifier.verify_use_case(use_case_id)
        
        print("\nClassification Verification Results:")
        print("-" * 80)
        print(f"Use Case: {results['use_case']['name']}")
        print(f"Purpose and Benefits: {results['use_case']['purpose_benefits']}")
        print("-" * 80)
        
        if results["classifications"]:
            print("\nClassifications Found:\n")
            for classification in results["classifications"]:
                print(f"Match Type: {classification['match_type']}")
                print(f"Category: {classification['category']}")
                print(f"Confidence: {classification['confidence']:.2f}")
                print(f"Reasoning: {classification['reasoning']}")
                print(f"Classified At: {classification['classified_at']}")
                print(f"Classified By: {classification['classified_by']}")
                print()
                
        if results["no_match_analysis"]:
            print("\nNo Match Analysis:")
            no_match = results["no_match_analysis"]
            print(f"Reason: {no_match['reason']}")
            print(f"LLM Analysis: {no_match['llm_analysis']}")
            print(f"Suggested Keywords: {no_match['suggested_keywords']}")
            print(f"Technical Gaps: {no_match['technical_gaps']}")
            print(f"Suggested Focus: {no_match['suggested_focus']}")
            print(f"Timestamp: {no_match['timestamp']}")
                
        print("\nVerification Complete!")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        sys.exit(1)
    finally:
        verifier.close()

if __name__ == "__main__":
    main() 