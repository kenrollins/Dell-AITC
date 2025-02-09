# Schema Refactoring Project 2024

## Overview
This document tracks the progress and decisions made during the 2024 schema refactoring effort for the Dell-AITC project.

## Current State (as of Feb 8, 2024)

### Project Phase: Schema Restructuring
```yaml
Status:
- Created deprecated structure for v1 content
- Moved initial schema-independent scripts
- Identified core scripts to maintain
- Simplified AI categories structure
```

### Directory Structure
```
Dell-AITC/
├── deprecated/v1/           # Legacy content
│   ├── scripts/            # Old script versions
│   ├── docs/              # Old documentation
│   └── data/              # Old data structures
├── docs/
│   ├── neo4j/             # Current schema documentation
│   ├── project_tracking/  # Project management docs
│   └── fed_use_case/     # Use case documentation
└── scripts/               # Current active scripts
```

### Key Files Status
```yaml
Maintained (Current):
- manage.py: Project management script
- database/init/reset.py: Database management
- tests/conftest.py: Test configuration

Deprecated (v1):
- check_evaluations.py: Moved to deprecated
- Other schema-dependent scripts: Pending review

Pending Creation:
- New schema visualization generator
- Updated data importers
- New classifier implementation
```

### Data Sources
```yaml
Current:
- AI-Technology-Categories-v1.4.csv
- AI-Technology-zones-v1.4.csv

Pending:
- 2024 AI Inventory data (to be reviewed)
```

## Schema Decisions

### Current Decisions
```yaml
Keyword Structure:
- Simplified to three types:
  1. technical_keywords
  2. capabilities
  3. business_language
- Removed input/output data types

Domain Separation:
- AI Technology Categories
- Federal Use Cases
- Classification
- Unmatched Analysis
- Schema Metadata

Relationships:
- Formalized zone relationships
- Clear parent-child structures
- Improved validation rules
```

### Pending Decisions
```yaml
1. Final keyword structure details
2. Relationship cardinality rules
3. Migration strategy for existing data
4. New script organization
5. 2024 AI inventory integration
```

## Next Steps

### Immediate Tasks
1. Review 2024 AI inventory data
2. Finalize schema structure
3. Create new schema visualization generator
4. Update core data importers

### Future Tasks
1. Implement new classifier
2. Create migration scripts
3. Update documentation
4. Create validation tools

## Requirements

### Core Requirements
```yaml
Functionality:
- Maintain backward compatibility where possible
- Clear separation of concerns
- Support for 2024 AI inventory
- Improved validation and constraints

Technical:
- Neo4j best practices
- Clear documentation
- Automated testing
- Data quality rules
```

### Migration Requirements
```yaml
Data:
- Preserve existing relationships
- Maintain data integrity
- Clear migration path
- Rollback capability

Scripts:
- Version compatibility
- Clear deprecation path
- Documentation updates
- Test coverage
```

## Migration Tools

### Schema Version Management
```yaml
Location: scripts/database/maintenance/schema_version_manager.py

Purpose:
- Manage schema version updates
- Create backups of existing schema files
- Replace old schema files with new versions
- Generate schema JSON from documentation
- Validate schema file consistency

Usage:
python scripts/database/maintenance/schema_version_manager.py

Process:
1. Creates timestamped backup of current schema
2. Copies new version files to root
3. Generates new schema JSON
4. Validates file consistency
```

## Notes
- This document should be updated as decisions are made
- All major changes should be documented here
- Use this as reference for new chat sessions
- Track all breaking changes

## References
- [SCHEMA_MASTER.md](/docs/neo4j/schema_planning/SCHEMA_MASTER.md)
- [Original Schema (Deprecated)](/deprecated/v1/docs/neo4j_schema_documentation.md) 