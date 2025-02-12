#!/usr/bin/env python3
"""
Live test for AI technology classification against Neo4j database.
Tests a specified number of random use cases with LLM analysis.

Usage:
    python test_live_classifier.py --n 1  # Test with 1 random use case
    python test_live_classifier.py --n 5  # Test with 5 random use cases
"""

import sys
import os
import logging
from pathlib import Path
import json
import asyncio
import argparse
from typing import Dict, Any, List

# Add the backend directory to Python path
backend_dir = str(Path(__file__).parent.parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.classifier import Classifier
from app.services.llm_analyzer import LLMAnalyzer
from app.models.analysis import AnalysisMethod, MatchType
from app.models.ai_category import AICategory
from app.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_no_match(use_case_id: str, use_case_name: str, scores: Dict[str, float], category: AICategory) -> Dict[str, Any]:
    """Analyze cases with no strong matches"""
    logging.info(f"Performing detailed analysis for low confidence match: {use_case_name}")
    
    analysis = {
        "use_case_id": use_case_id,
        "use_case_name": use_case_name,
        "category": category.name,
        "match_type": "NONE",
        "confidence": scores.get("combined_score", 0.0) * 100,
        "scores": {
            "keyword_score": scores.get("keyword_score", 0.0) * 100,
            "semantic_score": scores.get("semantic_score", 0.0) * 100,
            "llm_score": scores.get("llm_score", 0.0) * 100
        },
        "category_details": {
            "definition": category.definition,
            "maturity_level": category.maturity_level,
            "zone": category.zone
        },
        "improvement_suggestions": [
            "Consider adding more specific keywords",
            "Review category definition alignment",
            "Add more detailed use case description"
        ]
    }
    
    return analysis

def format_results(use_case_id: str, use_case_name: str, category_name: str, match_type: str, confidence: float, results: Dict[str, Any]) -> Dict[str, Any]:
    """Format classification results for display"""
    return {
        "use_case": {
            "id": use_case_id,
            "name": use_case_name
        },
        "classification": {
            "category_name": category_name,
            "match_type": match_type,
            "confidence": f"{confidence * 100:.1f}%",
            "scores": {
                "keyword_score": f"{results.get('keyword_score', 0.0) * 100:.1f}%",
                "semantic_score": f"{results.get('semantic_score', 0.0) * 100:.1f}%",
                "llm_score": f"{results.get('llm_score', 0.0) * 100:.1f}%"
            }
        }
    }

async def analyze_use_case(classifier: Classifier, use_case: Dict[str, Any]) -> None:
    """Analyze a single use case and display results"""
    logger.info(f"\nAnalyzing Use Case: {use_case['id']} - {use_case['name']}")
    
    try:
        # Perform initial classification
        logger.info("Performing initial classification...")
        results = await classifier.classify_use_case(use_case)
        
        # Format and display results
        formatted_results = format_results(
            results['use_case_id'], 
            results['use_case_name'], 
            results['category_name'], 
            results['match_type'], 
            results['confidence'], 
            results
        )
        logger.info("\nClassification Results:")
        logger.info(json.dumps(formatted_results, indent=2))
        
        # Perform additional analysis for no matches or low confidence
        if (results.get('match_type') == MatchType.NO_MATCH.value or 
            results.get('confidence', 0.0) < 0.3):
            
            logger.info("\nPerforming detailed analysis for low confidence match...")
            llm_results = await classifier.analyze_no_match(use_case, results.get('category_name'))
            
            logger.info("\nDetailed Analysis Results:")
            logger.info(json.dumps({
                'Suggested Category': llm_results['category'],
                'Confidence': f"{llm_results['confidence']*100:.1f}%",
                'Explanation': llm_results['explanation'],
                'Improvement Suggestions': llm_results['suggestions']
            }, indent=2))
            
    except Exception as e:
        logger.error(f"Error analyzing use case {use_case['id']}: {str(e)}")
        raise

async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test the classifier with live data.')
    parser.add_argument('--n', type=int, default=1, help='Number of use cases to test')
    args = parser.parse_args()

    logger.info(f"Initializing test with {args.n} random use cases...")
    
    classifier = Classifier(dry_run=True)
    
    try:
        # Initialize the classifier
        await classifier.initialize()
        logger.info("Classifier initialized successfully")
        
        # Get random use cases
        use_cases = await classifier.get_random_use_cases(args.n)
        logger.info(f"Found {len(use_cases)} use cases to analyze")
        
        # Process each use case
        for use_case in use_cases:
            await analyze_use_case(classifier, use_case)
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise
        
    finally:
        await classifier.close()

if __name__ == "__main__":
    asyncio.run(main()) 