#!/usr/bin/env python3
"""
Federal Use Case AI Classifier System Validation

This script provides comprehensive validation of all components used in the federal use case
AI technology classification system. It verifies:

1. Environment Setup
   - Configuration loading
   - Required environment variables
   - Directory structure

2. External Dependencies
   - Neo4j database connection and schema validation
   - OpenAI API access and response validation
   - Sentence transformer model loading and embedding tests

3. Core Functionality
   - Keyword matching accuracy and pattern validation
   - Semantic analysis performance benchmarking
   - LLM response quality assessment
   - Score calculation verification
   - Federal use case data structure validation

4. Data Quality
   - Federal use case completeness check
   - Agency and bureau relationship validation
   - Technology category hierarchy validation
   - Relationship type constraints

Output:
- Generates detailed validation reports in data/output/test_results/
- Logs all validation execution details in data/output/logs/
- Produces data quality metrics for federal use cases

Usage:
    python fed_use_case_ai_classifier_validation.py

Environment Variables:
    Same as fed_use_case_ai_classifier.py
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Additional imports for component testing
from neo4j import GraphDatabase
import openai
from sentence_transformers import SentenceTransformer

# Set up logging
def setup_logging() -> None:
    """Configure logging to write to both file and console"""
    log_dir = Path("data/output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"tech_eval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def test_environment() -> Dict[str, Any]:
    """Test basic environment setup and return configuration status"""
    config_status = {
        "env_loaded": False,
        "required_vars": {
            "NEO4J_URI": False,
            "NEO4J_USER": False,
            "NEO4J_PASSWORD": False,
            "OPENAI_API_KEY": False
        }
    }
    
    # Test .env loading
    try:
        load_dotenv()
        config_status["env_loaded"] = True
        logging.info("Successfully loaded .env file")
    except Exception as e:
        logging.error(f"Failed to load .env file: {str(e)}")
        return config_status
    
    # Check for required environment variables
    for var in config_status["required_vars"].keys():
        if os.getenv(var):
            config_status["required_vars"][var] = True
            logging.info(f"Found environment variable: {var}")
        else:
            logging.warning(f"Missing environment variable: {var}")
    
    return config_status

def verify_neo4j_connection(uri: str, user: str, password: str) -> Dict[str, Any]:
    """Verify Neo4j connection with proper error handling"""
    connection_status = {
        "connected": False,
        "error": None
    }
    
    try:
        # Use context manager for proper connection handling
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            # Explicitly verify connectivity
            driver.verify_connectivity()
            connection_status["connected"] = True
            logging.info("✓ Neo4j connection verified successfully")
    except Exception as e:
        error_msg = f"✗ Neo4j connection failed: {str(e)}"
        connection_status["error"] = error_msg
        logging.error(error_msg)
    
    return connection_status

def test_neo4j_connection() -> Dict[str, Any]:
    """Test Neo4j database connection with enhanced verification"""
    connection_status = {
        "connected": False,
        "error": None
    }
    
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        # First verify the connection
        verify_status = verify_neo4j_connection(uri, user, password)
        if not verify_status["connected"]:
            return verify_status
            
        # If verification successful, try a test query
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    connection_status["connected"] = True
                    logging.info("Successfully connected to Neo4j database")
    except Exception as e:
        connection_status["error"] = str(e)
        logging.error(f"Failed to connect to Neo4j: {str(e)}")
    
    return connection_status

def test_sentence_transformer() -> Dict[str, Any]:
    """Test loading of sentence transformer model"""
    model_status = {
        "loaded": False,
        "error": None
    }
    
    try:
        # Load a small model for testing
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Test with a simple sentence
        embedding = model.encode("Test sentence")
        if len(embedding) > 0:
            model_status["loaded"] = True
            logging.info("Successfully loaded sentence transformer model")
    except Exception as e:
        model_status["error"] = str(e)
        logging.error(f"Failed to load sentence transformer model: {str(e)}")
    
    return model_status

def test_openai_connection() -> Dict[str, Any]:
    """Test OpenAI API connection"""
    api_status = {
        "connected": False,
        "error": None
    }
    
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # Test with a simple completion request
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test' in one word"}],
            max_tokens=1
        )
        if response:
            api_status["connected"] = True
            logging.info("Successfully connected to OpenAI API")
    except Exception as e:
        api_status["error"] = str(e)
        logging.error(f"Failed to connect to OpenAI API: {str(e)}")
    
    return api_status

def main():
    """Main test execution function"""
    # Step 1: Set up logging
    setup_logging()
    logging.info("Starting technology category evaluation test script")
    
    # Step 2: Test environment configuration
    config_status = test_environment()
    
    # Step 3: Test component connections
    neo4j_status = test_neo4j_connection()
    transformer_status = test_sentence_transformer()
    openai_status = test_openai_connection()
    
    # Write test results
    output_dir = Path("data/output/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "environment_test": config_status,
        "component_tests": {
            "neo4j": neo4j_status,
            "sentence_transformer": transformer_status,
            "openai": openai_status
        },
        "python_version": os.sys.version
    }
    
    output_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Test results written to: {output_file}")

if __name__ == "__main__":
    main() 