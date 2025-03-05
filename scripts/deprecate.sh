#!/bin/bash

# Set project root
export PROJECT_ROOT="/data/anaconda/Dell-AITC"

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 \"REASON\" \"REPLACEMENT\" FILE1 [FILE2 ...]"
    echo "Example: $0 \"Outdated implementation\" \"new_script.py\" old_script.py"
    exit 1
fi

# Extract reason and replacement
REASON="$1"
REPLACEMENT="$2"
shift 2

# Run the manage_deprecated.py script
python scripts/manage_deprecated.py --reason "$REASON" --replacement "$REPLACEMENT" "$@"

# Show what was moved
echo ""
echo "Files moved to deprecated/scripts/:"
for file in "$@"; do
    echo "  - $(basename $file)"
done

echo ""
echo "To find more candidates for deprecation, run:"
echo "./scripts/find_deprecated.sh [directory]" 