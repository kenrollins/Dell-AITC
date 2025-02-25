#!/bin/bash

# Set project root
export PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Dell-AITC

# Set up environment variables for Neo4j
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="kuxFc8HN"

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

echo "Starting database reset process..."

# Set PYTHONPATH to include backend directory
export PYTHONPATH="${PROJECT_ROOT}/backend"

# Unset PYTHONHOME to avoid conflicts
unset PYTHONHOME

# Run the recovery script
python -m app.services.database.management.recover_from_nuke

echo "Database reset process completed." 