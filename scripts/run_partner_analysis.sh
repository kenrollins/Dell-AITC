#!/bin/bash

# Set project root
export PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Dell-AITC

# Set up environment variables for Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="kuxFc8HN"

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Unset problematic Python variables
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONSTARTUP

# Run the partner_analysis.py script with the provided arguments
python scripts/partner_analysis.py "$@" 