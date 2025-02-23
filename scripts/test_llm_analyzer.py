"""
Test script for LLM-based AI technology classification
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, List

from backend.app.services.ai.llm_analyzer import LLMAnalyzer
from backend.app.services.database.neo4j_service import Neo4jService
from backend.app.config import get_settings

# Configure logging
log_dir = Path("logs/tests")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'llm_analyzer_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_single_use_case():
    """Test LLM analyzer with a single use case"""
    try:
        # Initialize services
        neo4j_service = Neo4jService()
        analyzer = LLMAnalyzer()
        await analyzer.initialize()
        
        # Get categories
        categories = await neo4j_service.get_all_categories()
        logger.info(f"Loaded {len(categories)} categories")
        
        # Get a test use case
        test_cases = await neo4j_service.get_test_cases(limit=1)
        if not test_cases:
            logger.error("No test cases found")
            return
            
        use_case = test_cases[0]
        
        # Log use case details
        logger.info("\nAnalyzing Use Case:")
        logger.info(f"Name: {use_case.get('name')}")
        logger.info(f"Description: {use_case.get('description')}")
        
        # Run analysis
        logger.info("\nRunning LLM analysis...")
        result = await analyzer.analyze_use_case(use_case, categories)
        
        # Save results
        output_dir = Path("data/output/tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"llm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "use_case": use_case,
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
            
        # Log results
        logger.info("\nAnalysis Results:")
        if "primary_match" in result:
            primary = result["primary_match"]
            logger.info(f"\nPrimary Match: {primary['category']}")
            logger.info(f"Confidence: {primary['confidence']:.2f}")
            logger.info(f"Reasoning: {primary['reasoning']}")
            
            if "technical_alignment" in primary:
                logger.info("\nTechnical Alignment:")
                tech = primary["technical_alignment"]
                logger.info(f"Matched Terms: {', '.join(tech.get('matched_terms', []))}")
                logger.info(f"Matched Capabilities: {', '.join(tech.get('matched_capabilities', []))}")
                logger.info(f"Zone Compatibility: {tech.get('zone_compatibility')}")
        
        if result.get("supporting_matches"):
            logger.info("\nSupporting Matches:")
            for match in result["supporting_matches"]:
                logger.info(f"- {match['category']} (Confidence: {match['confidence']:.2f})")
                logger.info(f"  Reasoning: {match['reasoning']}")
                
        if result.get("related_matches"):
            logger.info("\nRelated Matches:")
            for match in result["related_matches"]:
                logger.info(f"- {match['category']} (Confidence: {match['confidence']:.2f})")
                logger.info(f"  Reasoning: {match['reasoning']}")
                
        if "no_match_analysis" in result:
            logger.info("\nNo Match Analysis:")
            no_match = result["no_match_analysis"]
            logger.info(f"Reason: {no_match['reason']}")
            logger.info(f"Technical Gaps: {', '.join(no_match['technical_gaps'])}")
            logger.info(f"Suggested Focus: {no_match['suggested_focus']}")
            logger.info("Improvement Suggestions:")
            for suggestion in no_match["improvement_suggestions"]:
                logger.info(f"- {suggestion}")
                
        logger.info(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise
        
    finally:
        # Cleanup
        await analyzer.cleanup()
        await neo4j_service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_single_use_case()) 