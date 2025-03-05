#!/bin/bash

# Set project root
export PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Default directory to scan
DIRECTORY=${1:-scripts}

# Run the identify_deprecated_candidates.py script
echo "Scanning directory: ${DIRECTORY}"
python scripts/identify_deprecated_candidates.py --directory "${DIRECTORY}"

# Provide instructions for using manage_deprecated.py
echo ""
echo "To move files to deprecated directory, use the manage_deprecated.py script:"
echo "python scripts/manage_deprecated.py --reason \"Your reason here\" --replacement \"Replacement script\" path/to/script.py"
echo ""
echo "Example:"
echo "python scripts/manage_deprecated.py --reason \"Outdated implementation\" --replacement \"new_script.py\" old_script.py" 