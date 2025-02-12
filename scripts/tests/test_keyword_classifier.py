#!/usr/bin/env python3
"""
Test script for keyword-based AI technology classification

This script tests the keyword matching implementation by:
1. Running test cases with known classifications
2. Comparing results with expected outcomes
3. Analyzing match quality and confidence scores
4. Generating detailed match analysis reports
"""

import sys
import os
from pathlib import Path
import json
import csv
from typing import Dict, List, Any, Tuple
import logging
import traceback
import argparse
import tqdm  # For progress bars
from datetime import datetime
from collections import Counter
import re
import asyncio

# Add project root and backend to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from app.services.classifier import Classifier

# Disable sentence transformer progress bars
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
tqdm.tqdm = lambda x, *args, **kwargs: x  # Disable tqdm progress bars

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# Configure root logger first
logging.basicConfig(
    level=logging.INFO,  # Change default level to INFO
    format='%(message)s',  # Simplified format
    handlers=[
        logging.FileHandler('classifier_test.log'),  # File handler first
        logging.StreamHandler(sys.stdout)  # Console handler second
    ],
    force=True  # Override any existing logger
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Add file handler for debug logs
debug_handler = logging.FileHandler('classifier_test_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

# Disable other loggers
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

# File paths
CATEGORIES_FILE = Path('data/input/AI-Technology-Categories-v1.4.csv')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test keyword-based AI technology classification')
    parser.add_argument(
        '-n', '--num-cases',
        type=int,
        default=10,
        help='Number of test cases to run (default: 10)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['keyword', 'semantic', 'llm', 'all'],
        default='all',
        help='Analysis method to use (default: all)'
    )
    return parser.parse_args()

def verify_paths():
    """Verify all required paths exist"""
    base_path = Path(__file__).parent.parent.parent
    paths_to_check = {
        'Backend Directory': base_path / 'backend',
        'Input Data Directory': base_path / 'data' / 'input',
        'AI Categories CSV': base_path / 'data' / 'input' / 'AI-Technology-Categories-v1.4.csv',
        'Use Cases CSV': base_path / 'data' / 'input' / '2024_consolidated_ai_inventory_raw_v2.csv'
    }
    
    for name, path in paths_to_check.items():
        exists = path.exists()
        logger.info(f"Checking {name}: {'✓' if exists else '✗'} {path}")
        if not exists:
            raise FileNotFoundError(f"{name} not found at: {path}")

def load_test_cases(limit: int = 10) -> List[Dict[str, Any]]:
    """Load test cases from CSV file"""
    test_cases = []
    csv_path = Path(__file__).parent.parent.parent / 'data' / 'input' / '2024_consolidated_ai_inventory_raw_v2.csv'
    
    logger.info(f"Loading test cases from: {csv_path} (limit: {limit})")
    
    # Column name mappings with variations of quotes
    column_variations = {
        'outputs': [
            "Describe the AI system's outputs.",
            "Describe the AI system\x92s outputs.",
            "Describe the AI system's outputs.",
            "Describe the AI systems outputs."
        ]
    }
    
    column_map = {
        'id': 'System Name',
        'name': 'Use Case Name',
        'description': 'Is the AI use case found in the below list of general commercial AI products and services?',
        'purpose_benefits': 'What is the intended purpose and expected benefits of the AI?',
        'outputs': None  # Will be set based on which variation is found
    }
    
    # Try different encodings
    encodings = ['latin1', 'cp1252', 'utf-8-sig', 'utf-8']
    last_error = None
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                logger.debug(f"Trying encoding: {encoding}")
                
                # Read and validate header
                header = f.readline().strip()
                logger.debug(f"Raw header: {header}")
                
                # Reset file pointer
                f.seek(0)
                
                reader = csv.DictReader(f)
                field_names = reader.fieldnames
                
                if not field_names:
                    logger.warning(f"No field names found with {encoding} encoding")
                    continue
                    
                logger.info(f"CSV columns ({len(field_names)}): {field_names}")
                
                # Find the correct output column name
                for variation in column_variations['outputs']:
                    if variation in field_names:
                        column_map['outputs'] = variation
                        break
                
                if not column_map['outputs']:
                    logger.warning(f"Could not find outputs column with {encoding}")
                    continue
                
                # Verify other required columns exist
                required_columns = [col for key, col in column_map.items() if col is not None]
                missing_columns = [col for col in required_columns if col not in field_names]
                if missing_columns:
                    logger.warning(f"Missing required columns with {encoding}: {missing_columns}")
                    continue
                
                for i, row in enumerate(reader):
                    if i >= limit:  # Use the specified limit
                        break
                    
                    try:
                        # Map columns and handle missing values
                        test_case = {
                            'id': row.get(column_map['id'], f'TEST_{i}'),
                            'name': row.get(column_map['name'], ''),
                            'description': row.get(column_map['description'], ''),
                            'purpose_benefits': row.get(column_map['purpose_benefits'], ''),
                            'outputs': row.get(column_map['outputs'], '')
                        }
                        
                        # Log raw values for debugging
                        logger.debug(f"Raw row {i+1}: {dict(row)}")
                        logger.debug(f"Mapped case {i+1}: {test_case}")
                        
                        # Only add cases with some content
                        if any(v.strip() for v in test_case.values()):
                            test_cases.append(test_case)
                            logger.debug(f"Loaded test case {len(test_cases)}: {test_case['name']}")
                    except Exception as e:
                        logger.error(f"Error processing row {i+1}: {str(e)}")
                        logger.debug(f"Row data: {row}")
                        continue
                
                if test_cases:
                    logger.info(f"Successfully loaded {len(test_cases)} test cases using {encoding} encoding")
                    return test_cases
                else:
                    logger.warning(f"No valid test cases found with {encoding} encoding")
                
        except UnicodeDecodeError as e:
            logger.debug(f"Failed to decode with {encoding}: {str(e)}")
            last_error = e
            continue
        except Exception as e:
            logger.error(f"Error loading test cases with {encoding}: {str(e)}")
            logger.debug(traceback.format_exc())
            last_error = e
            continue
    
    if not test_cases:
        error_msg = f"Could not load test cases with any encoding. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return test_cases

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze classification results"""
    analysis = {
        'total_cases': len(results),
        'matched_cases': 0,
        'unmatched_cases': 0,
        'confidence_distribution': {
            'high': 0,    # > 0.8
            'medium': 0,  # 0.5 - 0.8
            'low': 0      # < 0.5
        },
        'matched_categories': {},
        'keyword_stats': {
            'total_matches': 0,
            'avg_matches_per_case': 0,
            'most_common_keywords': {}
        }
    }
    
    for result in results:
        # Count matches vs non-matches
        best_match = result.get_best_match() if hasattr(result, 'get_best_match') else None
        
        if best_match:
            analysis['matched_cases'] += 1
            
            # Track categories
            cat = best_match.get('category_name')
            if cat:
                analysis['matched_categories'][cat] = analysis['matched_categories'].get(cat, 0) + 1
            
            # Track confidence levels
            conf = best_match.get('confidence', 0.0)
            if conf > 0.8:
                analysis['confidence_distribution']['high'] += 1
            elif conf > 0.5:
                analysis['confidence_distribution']['medium'] += 1
            else:
                analysis['confidence_distribution']['low'] += 1
                
            # Track keyword stats
            matched_terms = result.matched_terms if hasattr(result, 'matched_terms') else {}
            for category_terms in matched_terms.values():
                analysis['keyword_stats']['total_matches'] += len(category_terms)
                for term in category_terms:
                    analysis['keyword_stats']['most_common_keywords'][term] = \
                        analysis['keyword_stats']['most_common_keywords'].get(term, 0) + 1
        else:
            analysis['unmatched_cases'] += 1
    
    # Calculate averages
    if analysis['matched_cases'] > 0:
        analysis['keyword_stats']['avg_matches_per_case'] = \
            analysis['keyword_stats']['total_matches'] / analysis['matched_cases']
    
    return analysis

def analyze_unmatched_cases(results: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze unmatched cases to suggest potential new keywords"""
    # Common words to exclude
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Technical terms to look for (common patterns)
    patterns = [
        r'\b\w+\s+(system|platform|tool|solution|framework)\b',  # System/platform names
        r'\b(AI|ML|NLP|CV|IoT|API)\b',                         # Common acronyms
        r'\b\w+\s+(detection|analysis|processing|automation)\b', # Technical operations
        r'\b(3D|2D)\s+\w+\b',                                  # 3D/2D terms
        r'\b\w+\s+(learning|intelligence|analytics)\b',         # AI-related terms
        r'\b(autonomous|automated|intelligent)\s+\w+\b',        # Autonomous/automated systems
        r'\b\w+\s+(recognition|classification|identification)\b' # Recognition tasks
    ]
    
    keyword_suggestions = {
        'technical_terms': Counter(),
        'phrases': Counter(),
        'by_category': {}
    }
    
    for result, case in zip(results, test_cases):
        if not result['category_name']:  # Unmatched case
            # Combine all text fields
            text = ' '.join([
                case.get('name', ''),
                case.get('description', ''),
                case.get('purpose_benefits', ''),
                case.get('outputs', '')
            ]).lower()
            
            # Extract technical terms using patterns
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    term = match.group(0)
                    keyword_suggestions['technical_terms'][term] += 1
            
            # Extract 2-3 word phrases
            words = text.split()
            for i in range(len(words)-1):
                phrase = ' '.join(words[i:i+2])
                if not any(word in stop_words for word in phrase.split()):
                    keyword_suggestions['phrases'][phrase] += 1
                if i < len(words)-2:
                    phrase = ' '.join(words[i:i+3])
                    if not any(word in stop_words for word in phrase.split()):
                        keyword_suggestions['phrases'][phrase] += 1
    
    # Filter and sort suggestions
    keyword_suggestions['technical_terms'] = dict(sorted(
        {k: v for k, v in keyword_suggestions['technical_terms'].items() if v > 1}.items(),
        key=lambda x: x[1], reverse=True
    ))
    
    keyword_suggestions['phrases'] = dict(sorted(
        {k: v for k, v in keyword_suggestions['phrases'].items() if v > 1}.items(),
        key=lambda x: x[1], reverse=True
    ))
    
    return keyword_suggestions

def save_keyword_suggestions(analysis: Dict[str, Any], timestamp: str) -> Path:
    """Save keyword suggestions to file"""
    output_dir = Path('data/output/keyword_suggestions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'keyword_suggestions_{timestamp}.json'
    
    # Format suggestions
    suggestions = {
        'keyword_stats': analysis['keyword_stats'],
        'matched_categories': analysis['matched_categories'],
        'confidence_distribution': analysis['confidence_distribution'],
        'total_cases': analysis['total_cases'],
        'matched_cases': analysis['matched_cases'],
        'unmatched_cases': analysis['unmatched_cases']
    }
    
    with open(output_file, 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    return output_file

async def process_test_cases(test_cases: List[Dict[str, Any]], classifier: Classifier) -> List[Dict[str, Any]]:
    """Process test cases asynchronously"""
    results = []
    for i, case in enumerate(test_cases, 1):
        try:
            logger.info(f"\nProcessing case {i}:")
            logger.info(f"Name: {case.get('name', 'N/A')}")
            logger.info(f"Description: {case.get('description', 'N/A')[:100]}...")
            logger.info(f"Purpose/Benefits: {case.get('purpose_benefits', 'N/A')[:100]}...")
            logger.info(f"Outputs: {case.get('outputs', 'N/A')[:100]}...")
            
            result = await classifier.classify_use_case(case)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing case {i}: {str(e)}")
            logger.error(f"Case data: {case}")
            results.append({
                'error': str(e),
                'case': case
            })
    return results

async def async_main():
    """Async main function"""
    try:
        args = parse_args()
        verify_paths()
        
        # Initialize classifier
        logger.info("Initializing classifier...")
        classifier = Classifier(dry_run=True)
        await classifier.initialize()
        
        # Load test cases
        logger.info(f"Loading {args.num_cases} test cases...")
        test_cases = load_test_cases(args.num_cases)
        
        logger.info(f"\nProcessing {len(test_cases)} test cases...")
        results = await process_test_cases(test_cases, classifier)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Print summary
        print("\nClassification Results:")
        print(f"Total Cases: {analysis['total_cases']}")
        print(f"Matched Cases: {analysis['matched_cases']}")
        print(f"Unmatched Cases: {analysis['unmatched_cases']}")
        
        # Save results if needed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_keyword_suggestions(analysis, timestamp)
        
        # Cleanup
        await classifier.close()
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main entry point"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 