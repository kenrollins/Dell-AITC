#!/usr/bin/env python
"""
Schema Version Manager for Dell-AITC
Manages schema versioning and documentation updates.

Usage:
    python schema_version_manager.py [options]

Options:
    --major          Increment major version (X.0.0)
    --minor          Increment minor version (x.Y.0)
    --version X.Y.Z  Set specific version
    --yes, -y        Skip confirmation prompt

Examples:
    # Increment patch version (2.1.1 -> 2.1.2)
    python schema_version_manager.py

    # Increment minor version (2.1.1 -> 2.2.0)
    python schema_version_manager.py --minor

    # Increment major version (2.1.1 -> 3.0.0)
    python schema_version_manager.py --major

    # Set specific version
    python schema_version_manager.py --version 2.1.3

    # Skip confirmation prompt
    python schema_version_manager.py --yes

Version Numbering:
    MAJOR.MINOR.PATCH
    - MAJOR: Breaking changes
    - MINOR: New features, backward compatible
    - PATCH: Bug fixes, small changes

The script will:
1. Automatically detect current version
2. Create backups of existing schema files
3. Consolidate completed changes
4. Update master schema documentation
5. Reset working changes document
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaVersionManager:
    """Manages schema versioning and documentation.
    
    This class handles:
    - Version tracking and increments
    - Schema file backups
    - Change consolidation
    - Documentation updates
    - Working changes management
    """
    
    def __init__(self, project_root: str = None):
        """Initialize with project root directory.
        
        Args:
            project_root: Optional root directory path. Defaults to current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.docs_dir = self.project_root / 'docs'
        self.neo4j_dir = self.docs_dir / 'neo4j'
        self.schema_planning_dir = self.neo4j_dir / 'schema_planning'
        
        # Core schema files that must exist
        self.core_files = [
            'neo4j_schema_documentation.md',
            'neo4j_schema_visualization.md',
            'neo4j_schema.json'
        ]
        
    def get_current_version(self) -> Tuple[int, int, int]:
        """Get current schema version from working changes."""
        try:
            # First try to get from working changes
            for version_dir in self.schema_planning_dir.glob('v*'):
                if version_dir.is_dir():
                    working_changes = version_dir / 'WORKING_CHANGES.md'
                    if working_changes.exists():
                        with open(working_changes, 'r') as f:
                            content = f.read()
                            # Look for "Current Version: vX.Y.Z"
                            for line in content.split('\n'):
                                if 'Current Version:' in line:
                                    version_str = line.split('v')[-1].strip()
                                    major, minor, patch = map(int, version_str.split('.'))
                                    return major, minor, patch
                                    
            # Fallback to schema JSON
            schema_json = self.neo4j_dir / 'neo4j_schema.json'
            if schema_json.exists():
                with open(schema_json, 'r') as f:
                    data = json.load(f)
                    version_str = data.get('version', '0.0.0')
                    major, minor, patch = map(int, version_str.split('.'))
                    return major, minor, patch
                    
            return 0, 0, 0
            
        except Exception as e:
            logger.error(f"Failed to get current version: {str(e)}")
            return 0, 0, 0
            
    def increment_version(self, major: bool = False, minor: bool = False) -> str:
        """Increment version number based on flags."""
        current_major, current_minor, current_patch = self.get_current_version()
        
        if major:
            return f"{current_major + 1}.0.0"
        elif minor:
            return f"{current_major}.{current_minor + 1}.0"
        else:
            return f"{current_major}.{current_minor}.{current_patch + 1}"
            
    def create_backup(self, version: str) -> Path:
        """Create backup of current schema files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.neo4j_dir / 'backups' / f'v{version}_{timestamp}'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file in self.core_files:
            src = self.neo4j_dir / file
            if src.exists():
                shutil.copy2(src, backup_dir / file)
                
        return backup_dir
        
    def consolidate_changes(self, version: str) -> None:
        """Consolidate working changes into SCHEMA_CHANGES.md."""
        try:
            working_changes = self.schema_planning_dir / f'v{version}' / 'WORKING_CHANGES.md'
            schema_changes = self.schema_planning_dir / f'v{version}' / 'SCHEMA_CHANGES.md'
            
            if not working_changes.exists():
                logger.warning(f"No working changes file found at {working_changes}")
                return
                
            # Read working changes
            with open(working_changes, 'r') as f:
                changes_content = f.read()
                
            # Extract completed changes
            # TODO: Implement logic to extract completed changes
            
            # Append to schema changes
            with open(schema_changes, 'a') as f:
                f.write(f"\n\n## Version {version}\n")
                f.write("Status: Completed\n")
                f.write(f"Date: {datetime.now().strftime('%B %Y')}\n\n")
                f.write(changes_content)
                
            # Reset working changes
            self.reset_working_changes(version)
            
        except Exception as e:
            logger.error(f"Failed to consolidate changes: {str(e)}")
            raise
            
    def reset_working_changes(self, version: str) -> None:
        """Reset working changes document to template."""
        try:
            working_changes = self.schema_planning_dir / f'v{version}' / 'WORKING_CHANGES.md'
            
            template = f"""# Schema Working Changes v{version}.x

## Overview
This document tracks schema changes discovered during development and testing. 
Changes will be consolidated into the master schema documentation when finalized.

## Current Version: v{version}
Previous changes have been consolidated into SCHEMA_CHANGES.md.

## Working Changes
```yaml
# Template for new changes:
1. [Node/Relationship] [Addition/Modification/Removal]
   - Specific changes
   Reason: Why this change is needed
   Status: [Proposed/Implemented/Tested/Documented]
```

## Process
1. Add changes to this document as they are discovered
2. Test changes in isolation
3. When changes are validated:
   - Update version number (increment PATCH)
   - Move changes to SCHEMA_CHANGES.md
   - Update schema JSON and documentation
   - Update visualization if needed

## Validation Checklist
For each change:
- [ ] Change documented here
- [ ] Change tested in isolation
- [ ] Impact assessed
- [ ] Migration steps identified
- [ ] Documentation updates planned
- [ ] Code updates identified

## Notes
- Keep this document updated as we discover needed changes
- Use this as a working document before formalizing changes
- Track dependencies between changes
- Note any rollback considerations
"""
            
            with open(working_changes, 'w') as f:
                f.write(template)
                
        except Exception as e:
            logger.error(f"Failed to reset working changes: {str(e)}")
            raise
            
    def update_schema_files(self, new_version: str) -> None:
        """Update schema files with new version."""
        try:
            logger.info(f"Starting schema update to version {new_version}")
            
            # Create backup first
            backup_dir = self.create_backup(new_version)
            logger.info(f"Created backup at {backup_dir}")
            
            # Consolidate changes
            self.consolidate_changes(new_version)
            
            # Copy schema files
            version_dir = self.schema_planning_dir / f'v{new_version}'
            
            file_mapping = {
                f"SCHEMA_MASTER_V{new_version}.md": "neo4j_schema_documentation.md",
                f"SCHEMA_VISUALIZATION_V{new_version}.md": "neo4j_schema_visualization.md",
                f"schema/neo4j_schema_v{new_version}.json": "neo4j_schema.json"
            }
            
            for src_name, dest_name in file_mapping.items():
                src = version_dir / src_name
                if src.exists():
                    dest = self.neo4j_dir / dest_name
                    shutil.copy2(src, dest)
                    logger.info(f"Updated {dest_name}")
                else:
                    logger.warning(f"Source file not found: {src}")
                    
            # Validate updated files
            if not self.validate_schema_files():
                raise Exception("Schema validation failed after update")
                
            logger.info("Schema update completed successfully")
            
        except Exception as e:
            logger.error(f"Schema update failed: {str(e)}")
            raise
            
    def validate_schema_files(self) -> bool:
        """Validate that all required schema files exist and are valid."""
        try:
            for file in self.core_files:
                file_path = self.neo4j_dir / file
                if not file_path.exists():
                    logger.error(f"Missing schema file: {file}")
                    return False
                    
            # TODO: Add more validation as needed
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False

def main():
    """Main function to run schema version updates."""
    parser = argparse.ArgumentParser(description='Schema Version Manager')
    parser.add_argument('--major', action='store_true', help='Increment major version')
    parser.add_argument('--minor', action='store_true', help='Increment minor version')
    parser.add_argument('--version', type=str, help='Specify exact version (overrides increment flags)')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    try:
        manager = SchemaVersionManager()
        
        if args.version:
            new_version = args.version
        else:
            new_version = manager.increment_version(args.major, args.minor)
            
        if not args.yes:
            response = input(f"This will update schema files to version {new_version}. Continue? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Update cancelled by user.")
                return
                
        manager.update_schema_files(new_version)
        
    except Exception as e:
        logger.error(f"Schema update failed: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    main() 