#!/bin/bash

# Set the project root
PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Dell-AITC

# Ensure we're in the project root
cd $PROJECT_ROOT

# Unset problematic Python variables
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONSTARTUP

# Default to Word format unless specified otherwise
FORMAT=${1:-word}

# Check if we need to run the partner analysis script first
if [ -z "$(ls -A data/output/partner_analysis/partner_analysis_*.csv 2>/dev/null)" ]; then
  echo "No consolidated CSV file found. Running partner analysis with consolidation..."
  python scripts/partner_analysis.py --consolidate
fi

# Run the analysis
echo "Generating partner analysis report..."
python scripts/analyze_partner_results.py \
  --output-format $FORMAT \
  --output-dir data/output/partner_analysis

echo "Partner analysis report generation complete."
echo "Report saved to data/output/partner_analysis/partner_summary_*.${FORMAT}" 