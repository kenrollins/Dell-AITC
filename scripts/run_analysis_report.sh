#!/bin/bash

# Set project root
export PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Dell-AITC

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Unset problematic Python variables
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONSTARTUP

# Default to Word format if not specified
FORMAT=${1:-word}

# Run the analyze_partner_results.py script with the provided arguments
python scripts/analyze_partner_results.py --output-format $FORMAT "${@:2}"

echo "Report generation complete!"
echo "You can find the report in the data/output/partner_analysis directory." 