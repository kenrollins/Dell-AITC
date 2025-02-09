# Scripts Directory Structure

## Overview
This directory contains all scripts for managing, initializing, and analyzing the Dell-AITC system.

## Directory Structure

```
scripts/
├── database/                 # Database management scripts
│   ├── init/                # Initialization and reset scripts
│   │   ├── reset.py         # Nuclear reset option for Neo4j
│   │   ├── schema.py        # Schema initialization
│   │   └── constraints.py   # Database constraints and indexes
│   ├── loaders/             # Data loading scripts
│   │   ├── tech_categories.py
│   │   ├── federal_inventory.py
│   │   └── relationships.py
│   └── utils/               # Database utilities
│       ├── connection.py    # Connection management
│       ├── validation.py    # Schema validation
│       └── visualization.py # Schema visualization
├── analysis/                # Analysis scripts
│   ├── classifiers/         # AI classification scripts
│   ├── evaluators/          # Evaluation scripts
│   └── reporters/          # Reporting scripts
└── tests/                  # Test scripts
    ├── database/           # Database tests
    │   ├── test_reset.py   # Tests for database reset
    │   └── test_schema.py  # Tests for schema initialization
    ├── analysis/           # Analysis tests
    └── conftest.py         # pytest configuration

## Usage

### Database Management
- `database/init/reset.py`: Nuclear reset option for Neo4j database
  - Usage: `python -m scripts.database.init.reset`
  - Creates backup before reset
  - Clears all data while preserving structure
  - Verifies successful restart

### Testing
- All tests are located in the `tests/` directory
- Run tests using pytest: `pytest scripts/tests/`
- Each major component has its own test directory

## Development Guidelines

1. **Script Organization**
   - Place scripts in appropriate subdirectories based on functionality
   - Use clear, descriptive names for scripts
   - Include docstrings and type hints

2. **Testing**
   - Write tests for all new functionality
   - Place tests in corresponding test directories
   - Use pytest fixtures for shared resources

3. **Documentation**
   - Update this README when adding new scripts
   - Include usage examples in script docstrings
   - Document any new dependencies in requirements.txt

4. **Logging**
   - Use the logging module for all scripts
   - Follow established logging format
   - Include appropriate log levels

## Dependencies
See `requirements.txt` for full list of dependencies. 