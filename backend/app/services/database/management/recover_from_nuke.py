"""
Dell-AITC Database Recovery Script (v2.2)
Rebuilds the entire Neo4j database from scratch by running initialization 
and data loading scripts in the correct sequence.

Usage:
    python -m backend.app.services.database.management.recover_from_nuke

Environment Variables Required:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
"""

import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List

# Configure logging
log_dir = Path("logs/database")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'database_recovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_step(step_name: str, module_name: str) -> bool:
    """Run a database management step and log its output."""
    logger.info(f"\n{'='*80}\nStarting {step_name}...\n{'='*80}")
    
    try:
        # Using subprocess to capture real-time output
        process = subprocess.Popen(
            [sys.executable, '-m', module_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(line)
                
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"\n[SUCCESS] {step_name} completed successfully\n")
            return True
        else:
            logger.error(f"\n[FAILED] {step_name} failed with exit code {process.returncode}\n")
            return False
            
    except Exception as e:
        logger.error(f"\n[ERROR] {step_name} failed with error: {str(e)}\n")
        return False

def main():
    """Main recovery process."""
    try:
        logger.info("Starting database recovery process...")
        
        # Step 1: Initialize database schema
        if not run_step(
            "Database Initialization", 
            "backend.app.services.database.management.initialize_db"
        ):
            raise Exception("Database initialization failed")
            
        # Step 2: Load technology zones
        if not run_step(
            "Loading Technology Zones", 
            "backend.app.services.database.management.load_zones"
        ):
            raise Exception("Zone loading failed")
            
        # Step 3: Load AI categories
        if not run_step(
            "Loading AI Categories", 
            "backend.app.services.database.management.load_categories"
        ):
            raise Exception("Category loading failed")
            
        # Step 4: Load federal AI inventory
        if not run_step(
            "Loading Federal AI Inventory", 
            "backend.app.services.database.management.load_inventory"
        ):
            raise Exception("Inventory loading failed")
            
        # Step 5: Apply v2.2 relationship migrations
        if not run_step(
            "Applying v2.2 Relationship Migrations",
            "backend.app.services.database.management.migrations.v2_2_relationships"
        ):
            raise Exception("v2.2 relationship migration failed")
            
        # Step 6: Apply schema updates for keyword relevance
        if not run_step(
            "Applying Keyword Relevance Schema Updates",
            "backend.app.services.database.management.update_schema"
        ):
            raise Exception("Keyword relevance schema update failed")
            
        logger.info("\n[SUCCESS] Database recovery completed successfully!")
        
    except Exception as e:
        logger.error(f"\n[FAILED] Database recovery failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 