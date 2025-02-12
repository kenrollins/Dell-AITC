# Issue: Schema Version Manager Script Fixes Required

## Overview
The schema version manager script (`scripts/database/maintenance/schema_version_manager.py`) requires fixes to handle path resolution, environment setup, and file location expectations correctly.

## Current Issues

### 1. Path Resolution Problems
- Script fails to locate schema files using relative paths
- Unable to find `neo4j_schema_documentation.md` in expected location
- Directory structure assumptions don't match actual project layout

### 2. Version Management Issues
- Defaults to version 0.0.1 instead of detecting current version (2.1.2)
- Version increment logic may not properly read existing version numbers
- Version validation against schema files inconsistent

### 3. File Location Expectations
- Script expects files in hardcoded locations that don't match project structure
- Backup directory path resolution failing
- Working changes and schema changes file paths need updating

### 4. Environment Setup
- Script environment setup incomplete
- Missing proper logging configuration
- Path resolution not OS-agnostic

## Required Fixes

### Path Resolution
1. Update path resolution to use project root as base
2. Implement OS-agnostic path joining
3. Add configuration for file locations
4. Validate file existence before operations

### Version Management
1. Fix current version detection logic
2. Implement proper version parsing from schema files
3. Add validation for version consistency across files
4. Improve version increment logic

### File Operations
1. Update file location configuration
2. Implement proper backup directory creation
3. Add validation for file existence
4. Improve error handling for file operations

### Environment Setup
1. Add proper environment initialization
2. Implement robust logging configuration
3. Add configuration validation
4. Improve error reporting

## Implementation Plan

### Phase 1: Core Fixes
1. Fix path resolution
2. Update version management
3. Correct file location handling

### Phase 2: Improvements
1. Add configuration system
2. Improve logging
3. Enhance error handling

### Phase 3: Testing
1. Add unit tests
2. Add integration tests
3. Create test fixtures

## Technical Details

### Current Script Location
```
scripts/database/maintenance/schema_version_manager.py
```

### Expected File Structure
```
docs/
  neo4j/
    schema_planning/
      v2.1/
        SCHEMA_CHANGES.md
        WORKING_CHANGES.md
        schema/
          neo4j_schema_v2.1.json
          setup_schema_v2.1.cypher
    neo4j_schema_documentation.md
    neo4j_schema_visualization.md
```

### Required Dependencies
- pathlib
- logging
- yaml
- json

## Success Criteria
1. Script successfully detects current version
2. Properly increments version numbers
3. Correctly locates and updates all schema files
4. Creates backups before modifications
5. Validates schema consistency
6. Works across different operating systems

## Notes
- Consider simplifying the script's architecture
- Add better documentation
- Consider adding a configuration file
- Make the script more modular

## Related Files
- schema_version_manager.py
- neo4j_schema.json
- neo4j_schema_documentation.md
- SCHEMA_CHANGES.md
- WORKING_CHANGES.md

## Priority
Medium - Current manual process is working, but automation would improve efficiency

## Status
Open - Awaiting implementation

## Assigned To
TBD

## Created
Date: 2024-02
Version: 2.1.2 