# Database Reset Instructions

## Quick Start
To reset the database, simply run:
```bash
./scripts/reset_database.sh
```

## What the Reset Process Does
1. Loads the AI inventory from `data/input/2024_consolidated_ai_inventory_raw_v2.csv`
2. Applies v2.2 relationship migrations
3. Applies keyword relevance schema updates
4. Rebuilds the entire database from scratch

## Prerequisites
- Neo4j database must be running
- Conda environment `Dell-AITC` must be properly set up
- Input data file must exist at `data/input/2024_consolidated_ai_inventory_raw_v2.csv`

## Environment Variables
The script automatically sets up:
- `PYTHONPATH` to include the backend directory
- Neo4j connection details (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
- Unsets `PYTHONHOME` to avoid conflicts

## Troubleshooting
If you encounter issues:
1. Ensure Neo4j is running and accessible
2. Verify the conda environment is activated
3. Check that all required input files exist
4. Verify Neo4j credentials in `.env` match those in the reset script

## Manual Reset
If you need to run the reset manually:
```bash
cd /data/anaconda/Dell-AITC
export PYTHONPATH=/data/anaconda/Dell-AITC/backend
unset PYTHONHOME
python -m app.services.database.management.recover_from_nuke
``` 