#!/usr/bin/env python3
"""
Integration test for AI technology classification

This script tests the classifier functionality with:
1. Dry run mode for testing without database changes
2. Single or multiple use case classification
3. Detailed output of classification results
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
import sys

from backend.app.services.classifier import Classifier
from backend.app.models.analysis import AnalysisMethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test AI technology classification'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--use-case-id',
        type=str,
        help='Specific use case ID to classify'
    )
    group.add_argument(
        '--sample-size',
        type=int,
        help='Number of random use cases to classify'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['keyword', 'semantic', 'llm', 'all'],
        default='all',
        help='Analysis method to use (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving to database'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Save results to JSON file'
    )
    return parser.parse_args()

def get_sample_use_cases(classifier: Classifier, sample_size: int) -> List[str]:
    """Get a sample of use case IDs from the database"""
    with classifier.driver.session() as session:
        # Get a random sample of use cases that don't already have classifications
        query = """
        MATCH (u:UseCase)
        WHERE NOT EXISTS((u)-[:USES_TECHNOLOGY]->(:AICategory))
        WITH u, rand() as r
        ORDER BY r
        LIMIT $limit
        RETURN u.id as id
        """
        result = session.run(query, limit=sample_size)
        use_case_ids = [record['id'] for record in result]
        
        if not use_case_ids:
            # If no unclassified cases found, get any use cases
            query = """
            MATCH (u:UseCase)
            WITH u, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN u.id as id
            """
            result = session.run(query, limit=sample_size)
            use_case_ids = [record['id'] for record in result]
            
        logger.info(f"Found {len(use_case_ids)} use cases to process")
        return use_case_ids

def format_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format classification results for display"""
    if not results:
        return {"error": "No results found"}
        
    formatted = {
        'Use Case': results['use_case_name'],
        'ID': results['use_case_id'],
        'Classification': {
            'Category': results['category_name'] or 'No Match',
            'Confidence': f"{results['confidence']*100:.1f}%",
            'Method': results['match_method']
        },
        'Scores': {
            'Keyword': f"{results['keyword_score']*100:.1f}%",
            'Semantic': f"{results['semantic_score']*100:.1f}%",
            'LLM': f"{results['llm_score']*100:.1f}%"
        }
    }
    
    if results['matched_keywords']:
        formatted['Matched Keywords'] = results['matched_keywords']
    
    if results['llm_explanation']:
        formatted['LLM Explanation'] = results['llm_explanation']
        
    return formatted

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

def process_use_case(classifier: Classifier, use_case_id: str, method: AnalysisMethod) -> Optional[Dict[str, Any]]:
    """Process a single use case and return formatted results"""
    logger.info(f"Classifying use case {use_case_id}...")
    
    results = classifier.classify_by_use_case_number(
        use_case_id,
        method=method
    )
    
    if not results:
        logger.warning(f"No results found for use case {use_case_id}")
        return None
        
    return format_results(results)

def main():
    """Main execution"""
    args = parse_args()
    
    logger.info("Starting classification test...")
    logger.info(f"Mode: {'Dry Run' if args.dry_run else 'Production'}")
    logger.info(f"Method: {args.method}")
    
    # Initialize classifier
    try:
        classifier = Classifier(dry_run=args.dry_run)
        method = AnalysisMethod[args.method.upper()]
        all_results = []
        
        if args.use_case_id:
            # Single use case
            logger.info(f"Processing single use case: {args.use_case_id}")
            result = process_use_case(classifier, args.use_case_id, method)
            if result:
                all_results.append(result)
                
        elif args.sample_size:
            # Multiple use cases
            logger.info(f"Processing {args.sample_size} use cases...")
            use_case_ids = get_sample_use_cases(classifier, args.sample_size)
            
            if not use_case_ids:
                logger.error("No use cases found in database")
                sys.exit(1)
                
            logger.info(f"Found {len(use_case_ids)} use cases to process")
            
            for use_case_id in use_case_ids:
                result = process_use_case(classifier, use_case_id, method)
                if result:
                    all_results.append(result)
        
        # Display results
        if all_results:
            print("\nClassification Results:")
            for i, result in enumerate(all_results, 1):
                print(f"\nResult {i}/{len(all_results)}:")
                print(json.dumps(result, indent=2))
                
            # Calculate statistics
            total = len(all_results)
            matched = sum(1 for r in all_results if r['Classification']['Category'] != 'No Match')
            print(f"\nSummary:")
            print(f"Total processed: {total}")
            print(f"Successfully matched: {matched} ({matched/total*100:.1f}%)")
            
            # Save to file if requested
            if args.output_file:
                save_results(all_results, args.output_file)
                logger.info(f"Results saved to {args.output_file}")
        else:
            logger.error("No results found for any use cases")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise

if __name__ == "__main__":
    main() 