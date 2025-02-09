#!/usr/bin/env python
"""
Dell-AITC Management Script
Run common tasks and get guidance on project management.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        
    def run_tests(self, integration: bool = False, coverage: bool = False) -> int:
        """Run project tests."""
        logger.info("Running tests...")
        cmd = ["pytest", "tests/"]
        
        if not integration:
            cmd.append("-m")
            cmd.append("not integration")
            
        if coverage:
            cmd.extend(["--cov=scripts", "--cov-report=term-missing"])
            
        return subprocess.run(cmd, cwd=self.project_root).returncode
        
    def check_code_quality(self) -> int:
        """Run code quality checks."""
        logger.info("Checking code quality...")
        
        # Run black
        logger.info("Running Black formatter check...")
        black_result = subprocess.run(
            ["black", "--check", "."],
            cwd=self.project_root
        )
        
        # Run flake8
        logger.info("Running Flake8 linter...")
        flake8_result = subprocess.run(
            ["flake8", "--max-complexity=10", "--max-line-length=100", "."],
            cwd=self.project_root
        )
        
        # Run mypy
        logger.info("Running MyPy type checker...")
        mypy_result = subprocess.run(
            ["mypy", "."],
            cwd=self.project_root
        )
        
        return max(black_result.returncode, flake8_result.returncode, mypy_result.returncode)
        
    def format_code(self) -> int:
        """Format code with Black."""
        logger.info("Formatting code with Black...")
        return subprocess.run(
            ["black", "."],
            cwd=self.project_root
        ).returncode
        
    def reset_database(self) -> int:
        """Reset the Neo4j database."""
        logger.info("Resetting Neo4j database...")
        from database.init.reset import Neo4jDatabaseManager
        
        try:
            db_manager = Neo4jDatabaseManager()
            db_manager.reset_database()
            return 0
        except Exception as e:
            logger.error(f"Database reset failed: {str(e)}")
            return 1
            
    def show_help(self):
        """Show detailed help about available commands."""
        help_text = """
Dell-AITC Management Commands
===========================

Test Commands:
-------------
python manage.py test              # Run unit tests
python manage.py test --integration # Run all tests including integration
python manage.py test --coverage   # Run tests with coverage report

Code Quality:
------------
python manage.py check            # Run all code quality checks
python manage.py format           # Format code with Black

Database:
---------
python manage.py reset-db         # Reset Neo4j database (with backup)

Development Workflow:
-------------------
1. Before committing:
   - Run 'python manage.py format' to format code
   - Run 'python manage.py check' to verify quality
   - Run 'python manage.py test' to run unit tests

2. Before merging:
   - Run 'python manage.py test --integration' for full test suite
   - Run 'python manage.py test --coverage' to check test coverage

3. Database management:
   - Use 'python manage.py reset-db' when you need a fresh database
   - Always check the backup location printed in logs
"""
        print(help_text)

def main():
    parser = argparse.ArgumentParser(description="Dell-AITC project management script")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--integration", action="store_true", help="Include integration tests")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Code quality commands
    subparsers.add_parser("check", help="Run code quality checks")
    subparsers.add_parser("format", help="Format code with Black")
    
    # Database commands
    subparsers.add_parser("reset-db", help="Reset Neo4j database")
    
    # Help command
    subparsers.add_parser("help", help="Show detailed help")
    
    args = parser.parse_args()
    manager = ProjectManager()
    
    if args.command == "test":
        sys.exit(manager.run_tests(args.integration, args.coverage))
    elif args.command == "check":
        sys.exit(manager.check_code_quality())
    elif args.command == "format":
        sys.exit(manager.format_code())
    elif args.command == "reset-db":
        sys.exit(manager.reset_database())
    elif args.command == "help" or not args.command:
        manager.show_help()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 