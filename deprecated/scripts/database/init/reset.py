import os
import shutil
import subprocess
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Optional
import neo4j
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jDatabaseManager:
    def __init__(self, 
                 neo4j_base_dir: str = r"D:\Docker\neo4j",
                 bolt_port: int = 7687,
                 http_port: int = 7474,
                 username: str = "neo4j",
                 password: str = "kuxFc8HN"):
        """
        Initialize the Neo4j database manager.
        
        Args:
            neo4j_base_dir: Base directory where Neo4j data is stored
            bolt_port: Neo4j Bolt protocol port
            http_port: Neo4j HTTP interface port
            username: Neo4j username
            password: Neo4j password
        """
        self.neo4j_base_dir = Path(neo4j_base_dir)
        self.subdirs = ['data', 'logs', 'plugins', 'config']
        self.bolt_port = bolt_port
        self.http_port = http_port
        self.username = username
        self.password = password
        self.uri = f"bolt://localhost:{bolt_port}"
        
    def test_connection(self, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """Test the Neo4j connection with retries."""
        logger.info(f"Testing connection to Neo4j at {self.uri}")
        
        for attempt in range(max_retries):
            try:
                with GraphDatabase.driver(self.uri, auth=(self.username, self.password)) as driver:
                    with driver.session() as session:
                        result = session.run("RETURN 1 as test")
                        result.single()
                logger.info("Successfully connected to Neo4j")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to Neo4j after {max_retries} attempts")
                    logger.error(f"Last error: {str(e)}")
                    return False
        return False
        
    def create_backup(self) -> Optional[Path]:
        """Create a backup of the current Neo4j data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.neo4j_base_dir.parent / f"neo4j_backup_{timestamp}"
        
        try:
            if self.neo4j_base_dir.exists():
                logger.info(f"Creating backup at: {backup_dir}")
                shutil.copytree(self.neo4j_base_dir, backup_dir)
                return backup_dir
            else:
                logger.warning(f"Neo4j directory {self.neo4j_base_dir} does not exist, skipping backup")
                return None
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise
            
    def stop_neo4j(self):
        """Stop the Neo4j Docker container."""
        try:
            logger.info("Stopping Neo4j container...")
            # First try docker compose
            compose_result = subprocess.run(
                ["docker-compose", "-f", str(self.neo4j_base_dir / "docker-compose.yml"), "down"],
                capture_output=True,
                text=True
            )
            
            if compose_result.returncode != 0:
                # Fallback to direct container stop
                logger.info("Docker compose failed, trying direct container stop...")
                subprocess.run(["docker", "stop", "neo4j"], check=True)
                subprocess.run(["docker", "rm", "neo4j"], check=True)
                
            # Give it a moment to fully stop
            time.sleep(5)
            
            # Verify container is stopped
            ps_result = subprocess.run(["docker", "ps", "-q", "-f", "name=neo4j"], 
                                     capture_output=True, text=True)
            if ps_result.stdout.strip():
                raise Exception("Failed to stop Neo4j container")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Neo4j container: {str(e)}")
            raise

    def start_neo4j(self):
        """Start the Neo4j Docker container."""
        try:
            logger.info("Starting Neo4j container...")
            subprocess.run(
                ["docker-compose", "-f", str(self.neo4j_base_dir / "docker-compose.yml"), "up", "-d"],
                check=True
            )
            
            # Test connection with retries
            if not self.test_connection():
                raise Exception("Failed to establish connection to Neo4j after startup")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Neo4j container: {str(e)}")
            raise

    def clear_data(self):
        """Clear all Neo4j data directories while preserving docker-compose.yml."""
        try:
            logger.info("Clearing Neo4j data directories...")
            for subdir in self.subdirs:
                dir_path = self.neo4j_base_dir / subdir
                if dir_path.exists():
                    logger.info(f"Clearing {subdir} directory...")
                    # Remove directory contents but keep the directory
                    for item in dir_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
        except Exception as e:
            logger.error(f"Failed to clear data: {str(e)}")
            raise

    def reset_database(self):
        """
        Perform a complete database reset:
        1. Create backup
        2. Stop container
        3. Clear data
        4. Start container
        5. Verify connection
        """
        try:
            logger.info("Starting database reset process...")
            
            # Create backup
            backup_path = self.create_backup()
            if backup_path:
                logger.info(f"Backup created at: {backup_path}")
            
            # Stop Neo4j
            self.stop_neo4j()
            
            # Clear data
            self.clear_data()
            
            # Start Neo4j
            self.start_neo4j()
            
            logger.info("Database reset completed successfully!")
            if backup_path:
                logger.info(f"Backup available at: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database reset failed: {str(e)}")
            logger.error("Please check the logs and try again.")
            raise

def main():
    """Main function to run the database reset."""
    try:
        # Get confirmation from user
        response = input("This will reset the Neo4j database. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Reset cancelled by user.")
            return
        
        # Initialize and run reset
        db_manager = Neo4jDatabaseManager()
        db_manager.reset_database()
        
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 