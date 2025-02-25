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

# Retry Case 4 that previously failed
python scripts/llm_tech_classifier.py \
  --use-case-id "25f4e66b-1a28-4492-8bbc-2f004571c8f0" \
  --model command-r-plus:104b \
  --num-gpus 1 \
  --batch-size 1 \
  --ollama-url "http://localhost:11434" 