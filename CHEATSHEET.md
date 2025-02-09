# Dell-AITC Cheat Sheet

## Quick Commands

### Database
```bash
# Reset database (with backup)
python scripts/manage.py reset-db

# Update schema version
python scripts/database/maintenance/schema_version_manager.py
```

### Code Quality
```bash
# Format code
python scripts/manage.py format

# Check quality
python scripts/manage.py check
```

### Testing
```bash
# Quick test
python scripts/manage.py test

# Full test suite
python scripts/manage.py test --integration

# With coverage
python scripts/manage.py test --coverage
```

### Help
```bash
# Show all commands
python scripts/manage.py help
```

## Neo4j Quick Access
- Browser: http://localhost:7474
- Credentials: neo4j/kuxFc8HN
- Backup Location: D:\Docker\neo4j_backup_<timestamp>

## Schema Management

### File Locations
```yaml
Current Schema:
- docs/neo4j/neo4j_schema_documentation.md
- docs/neo4j/neo4j_schema_visualization.md
- docs/neo4j/neo4j_schema.json

Development:
- docs/neo4j/schema_planning/v2/SCHEMA_MASTER_V2.md
- docs/neo4j/schema_planning/v2/SCHEMA_VISUALIZATION_V2.md

Backups:
- docs/neo4j/backups/v1_<timestamp>/
```

### Schema Update Process
1. Review changes in schema_planning
2. Run schema version manager
3. Verify file consistency
4. Update applications

## Development Checklist

Before Commit:
- [ ] Format code: `manage.py format`
- [ ] Check quality: `manage.py check`
- [ ] Run tests: `manage.py test`

Before PR:
- [ ] Run all tests: `manage.py test --integration`
- [ ] Check coverage: `manage.py test --coverage`
- [ ] Update docs if needed
- [ ] Update schema if needed

## Common Issues

Database:
- Not connecting? → Check Docker is running
- Need fresh start? → Run reset-db command

Tests:
- Failing? → Run with -v flag
- Integration failing? → Check Neo4j is running

Schema:
- Missing files? → Check backup directory
- Update failed? → Check schema manager logs
- Need rollback? → Use backup from docs/neo4j/backups/ 

## Script Documentation Standards
Every Python script must include this documentation block:
```python
"""
Script Name and Purpose

Usage:
    python script_name.py [options]

Options:
    List all command-line flags and arguments

Examples:
    Practical examples of common use cases

Additional Info:
    Any other relevant details
"""
```

## Schema Version Management

### Quick Commands
```bash
# View current version and options
python scripts/database/maintenance/schema_version_manager.py --help

# Increment patch version (2.1.1 -> 2.1.2)
python scripts/database/maintenance/schema_version_manager.py

# Increment minor version (2.1.1 -> 2.2.0)
python scripts/database/maintenance/schema_version_manager.py --minor

# Increment major version (2.1.1 -> 3.0.0)
python scripts/database/maintenance/schema_version_manager.py --major

# Set specific version
python scripts/database/maintenance/schema_version_manager.py --version X.Y.Z

# Skip confirmation prompt
python scripts/database/maintenance/schema_version_manager.py --yes
```

### Version Types
- MAJOR: Breaking changes (X.0.0)
- MINOR: New features, backward compatible (x.Y.0)
- PATCH: Bug fixes, small changes (x.y.Z)

### Process Steps
1. Document changes in `docs/neo4j/schema_planning/vX.Y/WORKING_CHANGES.md`
2. Test changes in isolation
3. Run schema version manager to:
   - Auto-detect current version
   - Backup schema files
   - Consolidate changes to SCHEMA_CHANGES.md
   - Update master documentation
   - Reset working changes template

### Required Files
- `SCHEMA_VISUALIZATION_VX.X.md`: Visual schema representation
- `neo4j_schema_vX.X.json`: Schema definition
- `SCHEMA_CHANGES.md`: Change history
- `WORKING_CHANGES.md`: Current changes in progress 