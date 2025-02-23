"""
Test script for keyword-based AI technology classification using real data from Neo4j.
"""

import sys
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
import json
from pprint import pformat
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.ai.fed_use_case_classifier import FedUseCaseClassifier
from backend.app.services.ai.llm_service import LLMService
from backend.app.services.database.neo4j_service import Neo4jService
from backend.app.config import get_settings

# Load environment variables
load_dotenv()

# Configure logging
log_dir = Path("logs/tests")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'keyword_classifier_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_category_details(category: Dict[str, Any]) -> None:
    """Log detailed category information."""
    logger.info(f"\nCategory Details for: {category['name']}")
    logger.info(f"ID: {category['id']}")
    logger.info(f"Zone: {category['zone']}")
    logger.info(f"Maturity Level: {category['maturity_level']}")
    logger.info(f"Status: {category['status']}")
    logger.info(f"Version: {category['version']}")
    
    if category.get('keywords'):
        logger.info("\nTechnical Keywords:")
        for kw in category['keywords']:
            logger.info(f"  - {kw['name']} (Score: {kw.get('relevance_score', 0.5):.2f})")
    
    if category.get('business_terms'):
        logger.info("\nBusiness Terms:")
        for term in category['business_terms']:
            logger.info(f"  - {term['name']} (Score: {term.get('relevance_score', 0.5):.2f})")

def log_use_case_details(use_case: Dict[str, Any]) -> None:
    """Log detailed use case information."""
    logger.info(f"\nUse Case Details:")
    logger.info(f"Name: {use_case['name']}")
    logger.info(f"ID: {use_case['id']}")
    logger.info(f"Status: {use_case['status']}")
    logger.info(f"Development Stage: {use_case.get('dev_stage', 'Unknown')}")
    logger.info(f"Development Method: {use_case.get('dev_method', 'Unknown')}")
    
    logger.info("\nDescription:")
    logger.info(use_case.get('description', 'No description'))
    
    logger.info("\nPurpose & Benefits:")
    logger.info(use_case.get('purpose_benefits', 'No purpose/benefits specified'))
    
    logger.info("\nOutputs:")
    logger.info(use_case.get('outputs', 'No outputs specified'))

def log_match_details(analysis: Dict[str, Any], category: Dict[str, Any]) -> None:
    """Log detailed matching information."""
    logger.info(f"\nMatch Analysis for Category: {category['name']}")
    logger.info(f"Match Type: {analysis['match_type']}")
    logger.info(f"Overall Confidence: {analysis['confidence']:.3f}")
    
    logger.info("\nField Scores:")
    for field, score in analysis['field_match_scores'].items():
        logger.info(f"  {field}: {score:.3f}")
    
    if analysis['matched_terms'].get('technical'):
        logger.info("\nMatched Technical Terms:")
        for term in analysis['matched_terms']['technical']:
            logger.info(f"  - {term}")
            
    if analysis['matched_terms'].get('business'):
        logger.info("\nMatched Business Terms:")
        for term in analysis['matched_terms']['business']:
            logger.info(f"  - {term}")
    
    logger.info(f"\nJustification: {analysis['justification']}")
    
    if analysis.get('improvement_suggestions'):
        logger.info("\nImprovement Suggestions:")
        for suggestion in analysis['improvement_suggestions']:
            logger.info(f"  - {suggestion}")

async def run_classification_test() -> None:
    """Run classification test with a single use case."""
    settings = get_settings()
    
    # Initialize services
    neo4j_service = Neo4jService()
    llm_service = LLMService(api_key=settings.openai_api_key)
    classifier = FedUseCaseClassifier(neo4j_service, llm_service)
    
    logger.info("Initializing classification test...")
    
    # Get categories
    categories = await neo4j_service.get_all_categories()
    logger.info(f"Loaded {len(categories)} categories from Neo4j")
    
    # Get single test case
    test_cases = await neo4j_service.get_test_cases(limit=1)
    if not test_cases:
        logger.error("No test cases found")
        return
        
    use_case = test_cases[0]
    log_use_case_details(use_case)
    
    results = []
    for category in categories:
        log_category_details(category)
        
        # Run classification
        analysis = await classifier.classify_use_case(use_case, category)
        log_match_details(analysis, category)
        
        results.append({
            "category": category['name'],
            "analysis": analysis
        })
        
        # Add a small delay between categories
        await asyncio.sleep(0.5)
    
    # Save results
    output_dir = Path("data/output/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"classifier_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "use_case": use_case,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nTest results saved to: {output_file}")
    
    # Cleanup
    await llm_service.cleanup()
    await neo4j_service.cleanup()

if __name__ == "__main__":
    asyncio.run(run_classification_test()) 