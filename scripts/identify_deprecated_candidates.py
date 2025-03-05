#!/usr/bin/env python3
"""
Identify Deprecated Candidates Script

This script analyzes the codebase to identify potential candidates for deprecation.
It looks for scripts that:
1. Have backup or duplicate versions
2. Are not referenced elsewhere in the codebase
3. Have test_ prefix but are not in a tests directory
4. Have .bak or other backup extensions

Usage:
    python scripts/identify_deprecated_candidates.py [options]

Options:
    --directory DIR           Directory to scan (default: scripts)
    --output-format FORMAT    Output format (text or json, default: text)
    --verbose                 Enable verbose output
"""

import os
import sys
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Identify potential candidates for deprecation")
    parser.add_argument("--directory", type=str, default="scripts", 
                        help="Directory to scan (default: scripts)")
    parser.add_argument("--output-format", type=str, choices=["text", "json"], default="text",
                        help="Output format (text or json, default: text)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def find_files_with_backup_extensions(directory: str) -> List[str]:
    """Find files with backup extensions like .bak, .old, .backup, etc."""
    backup_extensions = [".bak", ".old", ".backup", ".tmp", ".temp", ".orig"]
    result = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in backup_extensions:
                result.append(os.path.join(root, file))
    
    return result

def find_duplicate_files(directory: str) -> List[Tuple[str, str]]:
    """Find potential duplicate files based on name similarity."""
    files = []
    duplicates = []
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.py', '.sh')):
                files.append(os.path.join(root, filename))
    
    # Compare file names without extensions
    file_bases = {}
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Handle common patterns like test_X.py and X.py
        if base_name.startswith("test_"):
            clean_name = base_name[5:]  # Remove "test_" prefix
        else:
            clean_name = base_name
        
        if clean_name in file_bases:
            duplicates.append((file_path, file_bases[clean_name]))
        else:
            file_bases[clean_name] = file_path
    
    return duplicates

def find_unreferenced_files(directory: str) -> List[str]:
    """Find files that are not referenced elsewhere in the codebase."""
    unreferenced = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.sh')) and not file.startswith('__'):
                file_path = os.path.join(root, file)
                
                # Skip if it's in a tests directory
                if "/tests/" in file_path:
                    continue
                
                # Check if the file is referenced elsewhere
                try:
                    # Use grep to search for references to this file
                    result = subprocess.run(
                        ["grep", "-r", "--include=*.py", "--include=*.sh", "--include=*.md",
                         os.path.basename(file), "."],
                        capture_output=True, text=True, check=False
                    )
                    
                    # Count references (excluding the file itself and imports within the same directory)
                    references = [line for line in result.stdout.splitlines() 
                                 if not line.startswith(file_path) and 
                                    not (os.path.dirname(file_path) in line and "import" in line)]
                    
                    if len(references) <= 1:  # Only self-reference or no references
                        unreferenced.append(file_path)
                except subprocess.SubprocessError:
                    print(f"Error checking references for {file_path}")
    
    return unreferenced

def find_test_scripts_outside_tests(directory: str) -> List[str]:
    """Find test scripts that are not in a tests directory."""
    test_scripts = []
    
    for root, _, files in os.walk(directory):
        # Skip if it's in a tests directory
        if "/tests/" in root:
            continue
            
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_scripts.append(os.path.join(root, file))
    
    return test_scripts

def main():
    """Main function."""
    args = parse_args()
    directory = args.directory
    verbose = args.verbose
    
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    print(f"Scanning directory: {directory}")
    
    # Find potential candidates for deprecation
    backup_files = find_files_with_backup_extensions(directory)
    duplicate_files = find_duplicate_files(directory)
    unreferenced_files = find_unreferenced_files(directory)
    test_scripts = find_test_scripts_outside_tests(directory)
    
    # Prepare results
    results = {
        "backup_files": backup_files,
        "duplicate_files": duplicate_files,
        "unreferenced_files": unreferenced_files,
        "test_scripts": test_scripts
    }
    
    # Output results
    if args.output_format == "json":
        # Convert tuple to list for JSON serialization
        json_results = {
            "backup_files": backup_files,
            "duplicate_files": [[a, b] for a, b in duplicate_files],
            "unreferenced_files": unreferenced_files,
            "test_scripts": test_scripts
        }
        print(json.dumps(json_results, indent=2))
    else:
        print("\n=== Potential Candidates for Deprecation ===\n")
        
        print("Files with backup extensions:")
        for file in backup_files:
            print(f"  - {file}")
        
        print("\nPotential duplicate files:")
        for file1, file2 in duplicate_files:
            print(f"  - {file1} and {file2}")
        
        print("\nUnreferenced files:")
        for file in unreferenced_files:
            print(f"  - {file}")
        
        print("\nTest scripts outside tests directories:")
        for file in test_scripts:
            print(f"  - {file}")
        
        print("\nTotal potential candidates:", 
              len(backup_files) + len(duplicate_files) + len(unreferenced_files) + len(test_scripts))
        
        print("\nTo move a file to the deprecated directory, use:")
        print("  mv <file_path> deprecated/scripts/")
        
        print("\nDon't forget to update deprecated/scripts/README.md with details about the deprecated files.")

if __name__ == "__main__":
    main() 