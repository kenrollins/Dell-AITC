# Scripts Directory

This directory contains utility scripts for the Dell-AITC project.

## Overview
This directory contains all scripts for managing, initializing, and analyzing the Dell-AITC system.

## Partner Analysis Scripts

### partner_analysis.py

Analyzes partners against AI technology categories and generates structured analysis results.

```bash
# Run analysis on all partners
python scripts/partner_analysis.py --output-format both

# Run analysis on a limited number of partners
python scripts/partner_analysis.py --limit 5 --output-format both

# Specify a different model
python scripts/partner_analysis.py --ollama-model deepseek-r1:70b
```

### analyze_partner_results.py

Generates comprehensive summary reports from partner analysis results.

```bash
# Generate a Word document report (recommended for presentations)
python scripts/analyze_partner_results.py --output-format word

# Generate a Markdown report
python scripts/analyze_partner_results.py --output-format markdown

# Specify a different input file
python scripts/analyze_partner_results.py --input-file path/to/consolidated.csv --output-format word
```

## Wrapper Scripts

### run_partner_analysis.sh

Wrapper script for running partner_analysis.py with proper environment setup.

```bash
# Run analysis on all partners
bash scripts/run_partner_analysis.sh --output-format both

# Run analysis on a limited number of partners
bash scripts/run_partner_analysis.sh --limit 5 --output-format both
```

### run_analysis_report.sh

Wrapper script for running analyze_partner_results.py with proper environment setup.

```bash
# Generate a Word document report (default)
bash scripts/run_analysis_report.sh

# Generate a Markdown report
bash scripts/run_analysis_report.sh markdown

# Specify additional options
bash scripts/run_analysis_report.sh word --min-confidence 0.8
```

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
├── partner_analysis.py     # Partner analysis script
├── analyze_partner_results.py # Partner analysis results summary script
├── run_partner_analysis.sh  # Wrapper for partner_analysis.py
├── run_analysis_report.sh   # Wrapper for analyze_partner_results.py
└── tests/                  # Test scripts
    ├── database/           # Database tests
    │   ├── test_reset.py   # Tests for database reset
    │   └── test_schema.py  # Tests for schema initialization
    ├── analysis/           # Analysis tests
    └── conftest.py         # pytest configuration
```

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

## Environment Setup

If you encounter issues with conda or Python environment, use the wrapper scripts which properly set up the environment before running the Python scripts.

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