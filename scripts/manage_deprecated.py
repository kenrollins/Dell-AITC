#!/usr/bin/env python3
"""
Manage Deprecated Files Script

This script helps manage deprecated files by:
1. Moving files to the deprecated directory
2. Updating the README.md in the deprecated directory
3. Providing a summary of what was moved

Usage:
    python scripts/manage_deprecated.py [options] file1 [file2 ...]

Options:
    --reason REASON           Reason for deprecation (required)
    --replacement REPLACEMENT Replacement file or functionality (required)
    --dry-run                 Show what would be done without making changes
    --verbose                 Enable verbose output

Example:
    python scripts/manage_deprecated.py --reason "Outdated implementation" --replacement "new_script.py" old_script.py
"""

import os
import sys
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manage deprecated files")
    parser.add_argument("files", nargs="+", help="Files to deprecate")
    parser.add_argument("--reason", type=str, required=True, help="Reason for deprecation")
    parser.add_argument("--replacement", type=str, required=True, help="Replacement file or functionality")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def ensure_deprecated_directory():
    """Ensure the deprecated/scripts directory exists."""
    os.makedirs("deprecated/scripts", exist_ok=True)

def move_file_to_deprecated(file_path: str, dry_run: bool = False, verbose: bool = False):
    """Move a file to the deprecated directory."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return False
    
    # Determine target path
    filename = os.path.basename(file_path)
    target_path = os.path.join("deprecated/scripts", filename)
    
    # Handle duplicate filenames
    if os.path.exists(target_path):
        base, ext = os.path.splitext(filename)
        target_path = os.path.join("deprecated/scripts", f"{base}.duplicate{ext}")
    
    if verbose or dry_run:
        print(f"Moving {file_path} to {target_path}")
    
    if not dry_run:
        shutil.move(file_path, target_path)
    
    return True

def update_readme(files: List[str], reason: str, replacement: str, dry_run: bool = False, verbose: bool = False):
    """Update the README.md in the deprecated directory."""
    readme_path = "deprecated/scripts/README.md"
    
    if not os.path.exists(readme_path):
        # Create a new README if it doesn't exist
        if verbose or dry_run:
            print(f"Creating new README at {readme_path}")
        
        if not dry_run:
            with open(readme_path, "w") as f:
                f.write("# Deprecated Scripts\n\n")
                f.write("This directory contains scripts that have been deprecated for various reasons. ")
                f.write("They are kept for reference but should not be used in production.\n\n")
                f.write("## Recently Deprecated Scripts\n\n")
                f.write("| Script | Deprecated Date | Reason | Replacement |\n")
                f.write("|--------|----------------|--------|-------------|\n")
    
    # Read the current README
    with open(readme_path, "r") as f:
        content = f.read()
    
    # Find the "Recently Deprecated Scripts" section
    recent_section_match = re.search(r"## Recently Deprecated Scripts\s*\n\s*\|.*\|\s*\n\s*\|[-\s|]*\|\s*\n", content)
    
    if not recent_section_match:
        print(f"Error: Could not find 'Recently Deprecated Scripts' section in {readme_path}")
        return False
    
    # Get the current date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Prepare new entries
    new_entries = ""
    for file_path in files:
        filename = os.path.basename(file_path)
        new_entries += f"| {filename} | {today} | {reason} | {replacement} |\n"
    
    # Insert new entries after the header
    section_end = recent_section_match.end()
    new_content = content[:section_end] + new_entries + content[section_end:]
    
    if verbose or dry_run:
        print(f"Updating {readme_path} with new entries:")
        print(new_entries)
    
    if not dry_run:
        with open(readme_path, "w") as f:
            f.write(new_content)
    
    return True

def main():
    """Main function."""
    args = parse_args()
    files = args.files
    reason = args.reason
    replacement = args.replacement
    dry_run = args.dry_run
    verbose = args.verbose
    
    if dry_run:
        print("DRY RUN: No changes will be made")
    
    # Ensure the deprecated directory exists
    ensure_deprecated_directory()
    
    # Move files to deprecated directory
    moved_files = []
    for file_path in files:
        if move_file_to_deprecated(file_path, dry_run, verbose):
            moved_files.append(file_path)
    
    if not moved_files:
        print("No files were moved")
        return
    
    # Update README
    if update_readme(moved_files, reason, replacement, dry_run, verbose):
        print(f"Successfully moved {len(moved_files)} files to deprecated/scripts/")
        if not dry_run:
            print("Updated deprecated/scripts/README.md")
    else:
        print("Failed to update README")

if __name__ == "__main__":
    main() 