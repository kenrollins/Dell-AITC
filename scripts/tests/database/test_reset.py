"""
Tests for the Neo4j database reset functionality.
"""

import pytest
from pathlib import Path
import docker
from unittest.mock import patch, MagicMock
from scripts.database.init.reset import Neo4jDatabaseManager

def test_create_backup(db_manager: Neo4jDatabaseManager, temp_neo4j_dir: Path):
    """Test backup creation functionality."""
    # Perform backup
    backup_path = db_manager.create_backup()
    
    assert backup_path is not None
    assert backup_path.exists()
    assert (backup_path / "data" / "dummy.db").exists()
    assert (backup_path / "docker-compose.yml").exists()

def test_clear_data(db_manager: Neo4jDatabaseManager, temp_neo4j_dir: Path):
    """Test data clearing functionality."""
    # Add some test files
    test_file = temp_neo4j_dir / "data" / "test.db"
    test_file.write_text("test data")
    
    # Clear data
    db_manager.clear_data()
    
    # Verify directories exist but are empty
    assert temp_neo4j_dir.exists()
    assert (temp_neo4j_dir / "data").exists()
    assert not (temp_neo4j_dir / "data" / "test.db").exists()

@pytest.mark.integration
def test_container_operations():
    """
    Test Docker container operations.
    Marked as integration test as it requires Docker.
    """
    # Use real Neo4j directory for Docker operations
    db_manager = Neo4jDatabaseManager()
    
    # Test stop operation
    db_manager.stop_neo4j()
    
    # Verify container is stopped
    client = docker.from_env()
    containers = client.containers.list(filters={"name": "neo4j"})
    assert len(containers) == 0
    
    # Test start operation
    db_manager.start_neo4j()
    
    # Verify container is running
    containers = client.containers.list(filters={"name": "neo4j"})
    assert len(containers) == 1
    assert containers[0].status == "running"

def test_connection_retry(db_manager: Neo4jDatabaseManager):
    """Test connection retry logic."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        # Setup mock to fail twice then succeed
        session_mock = MagicMock()
        session_mock.run.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            MagicMock(single=lambda: True)
        ]
        
        driver_mock = MagicMock()
        driver_mock.__enter__.return_value = driver_mock
        driver_mock.session.return_value = session_mock
        mock_driver.return_value = driver_mock
        
        # Test connection
        assert db_manager.test_connection(max_retries=3, retry_delay=0)
        
        # Verify we tried 3 times
        assert session_mock.run.call_count == 3

@pytest.mark.integration
def test_full_reset_workflow():
    """
    Test the complete reset workflow.
    Marked as integration test as it requires Docker.
    """
    db_manager = Neo4jDatabaseManager()
    
    try:
        # Perform reset
        db_manager.reset_database()
        
        # Verify connection works after reset
        assert db_manager.test_connection()
        
        # Verify container is running
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "neo4j"})
        assert len(containers) == 1
        assert containers[0].status == "running"
        
    except Exception as e:
        pytest.fail(f"Reset workflow failed: {str(e)}")

def test_backup_nonexistent_dir(tmp_path: Path):
    """Test backup behavior with nonexistent directory."""
    nonexistent_dir = tmp_path / "nonexistent"
    db_manager = Neo4jDatabaseManager(neo4j_base_dir=str(nonexistent_dir))
    
    # Should return None but not raise
    assert db_manager.create_backup() is None 