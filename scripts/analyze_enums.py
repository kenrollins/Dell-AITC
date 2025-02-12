"""
Analyze enum values in the AI inventory CSV file.
Compares actual values against expected valid values and outputs discrepancies.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import json
from typing import List, Dict

# Import the validation constants and mappings from load_inventory
from backend.app.services.database.management.load_inventory import (
    VALID_TOPIC_AREAS,
    VALID_STAGES,
    VALID_IMPACT_TYPES,
    VALID_DEV_METHODS,
    TOPIC_AREA_MAPPINGS,
    STAGE_MAPPINGS,
    IMPACT_TYPE_MAPPINGS,
    DEV_METHOD_MAPPINGS,
    clean_string
)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def analyze_enum_field(df: pd.DataFrame, field_name: str, valid_values: List[str], mappings: Dict[str, str]) -> Dict:
    """Analyze a single enum field with mapping support."""
    # Get actual values and their counts
    value_counts = df[field_name].value_counts()
    actual_values = value_counts.index.tolist()
    
    # Track different categories of values
    valid_matches = []
    mapped_matches = []
    invalid_values = []
    
    for value in actual_values:
        clean_value = clean_string(value)
        
        # Skip empty values
        if not clean_value:
            continue
            
        # Check direct matches
        is_valid = False
        for valid_value in valid_values:
            if clean_value == valid_value.lower():
                valid_matches.append((value, valid_value, int(value_counts[value])))
                is_valid = True
                break
                
        # Check mappings
        if not is_valid and clean_value in mappings:
            mapped_matches.append((value, mappings[clean_value], int(value_counts[value])))
            is_valid = True
            
        # Check semicolon-separated values
        if not is_valid and ';' in clean_value:
            parts = [p.strip() for p in clean_value.split(';')]
            for part in parts:
                if part in mappings:
                    mapped_matches.append((value, mappings[part], int(value_counts[value])))
                    is_valid = True
                    break
                for valid_value in valid_values:
                    if part == valid_value.lower():
                        valid_matches.append((value, valid_value, int(value_counts[value])))
                        is_valid = True
                        break
                if is_valid:
                    break
                    
        # If still invalid, add to invalid list
        if not is_valid:
            invalid_values.append((value, int(value_counts[value])))
    
    return {
        'field': field_name,
        'valid_matches': valid_matches,
        'mapped_matches': mapped_matches,
        'invalid_values': invalid_values,
        'total_records': int(len(df)),
        'unique_values': len(actual_values)
    }

def main():
    # Find the inventory file
    data_dir = Path("data/input")
    inventory_file = next(data_dir.glob("[0-9][0-9][0-9][0-9]_consolidated_ai_inventory_raw_v*.csv"))
    
    # Read CSV with different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(inventory_file, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not read file with any supported encoding")

    # Map CSV columns to our expected column names
    column_mapping = {
        'Use Case Topic Area': 'topic_area',
        'Stage of Development': 'stage',
        'Is the AI use case rights-impacting, safety-impacting, both, or neither?': 'impact_type',
        'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ': 'dev_method'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Analyze each enum field
    results = {
        'topic_area': analyze_enum_field(df, 'topic_area', VALID_TOPIC_AREAS, TOPIC_AREA_MAPPINGS),
        'stage': analyze_enum_field(df, 'stage', VALID_STAGES, STAGE_MAPPINGS),
        'impact_type': analyze_enum_field(df, 'impact_type', VALID_IMPACT_TYPES, IMPACT_TYPE_MAPPINGS),
        'dev_method': analyze_enum_field(df, 'dev_method', VALID_DEV_METHODS, DEV_METHOD_MAPPINGS)
    }
    
    # Save detailed results
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enum_analysis_detailed.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\nEnum Analysis Summary:")
    print("=====================")
    
    for field, result in results.items():
        print(f"\n{field.upper()}:")
        print(f"Total records: {result['total_records']}")
        print(f"Unique values: {result['unique_values']}")
        
        print("\nDirect matches:")
        for value, mapped_to, count in result['valid_matches']:
            print(f"  - {value} ({count} occurrences)")
            
        print("\nMapped values:")
        for value, mapped_to, count in result['mapped_matches']:
            print(f"  - {value} -> {mapped_to} ({count} occurrences)")
            
        if result['invalid_values']:
            print("\nInvalid values:")
            for value, count in result['invalid_values']:
                print(f"  - {value} ({count} occurrences)")
        else:
            print("\nNo invalid values found!")
            
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 