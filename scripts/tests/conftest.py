"""
Pytest configuration file for Dell-AITC tests.
Contains shared fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
import os
import tempfile
import shutil
from typing import Generator, Any

from scripts.database.init.reset import Neo4jDatabaseManager

@pytest.fixture(scope="session")
def neo4j_credentials() -> dict[str, Any]:
    """Provide Neo4j test credentials."""
    return {
        "username": "neo4j",
        "password": "kuxFc8HN",
        "bolt_port": 7687,
        "http_port": 7474
    }

@pytest.fixture(scope="function")
def temp_neo4j_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory structure mimicking Neo4j data directory.
    This allows tests to verify file operations without touching the real Neo4j installation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        neo4j_dir = Path(temp_dir) / "neo4j"
        # Create Neo4j directory structure
        for subdir in ['data', 'logs', 'plugins', 'config']:
            (neo4j_dir / subdir).mkdir(parents=True)
        
        # Create some dummy files
        (neo4j_dir / "docker-compose.yml").write_text("dummy compose file")
        (neo4j_dir / "data" / "dummy.db").write_text("dummy db file")
        
        yield neo4j_dir
        
        # Cleanup is handled by context manager

@pytest.fixture(scope="function")
def db_manager(temp_neo4j_dir: Path, neo4j_credentials: dict[str, Any]) -> Neo4jDatabaseManager:
    """Provide a Neo4jDatabaseManager instance configured for testing."""
    return Neo4jDatabaseManager(
        neo4j_base_dir=str(temp_neo4j_dir),
        **neo4j_credentials
    ) 