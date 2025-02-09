#!/usr/bin/env python3
"""
Transform Classifier Output

This script transforms the output from the classifier into the format expected by the importer.
It reads the classifier's detailed output CSV and creates a new CSV with the correct format.
"""

import pandas as pd
import sys
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transform_classifier_output')

def validate_and_fix_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and fix score columns to ensure they are between 0 and 1"""
    score_columns = ['keyword_score', 'semantic_score', 'llm_score', 'final_score', 'confidence']
    
    for col in score_columns:
        # Convert to float
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log any invalid values
        invalid_mask = ~df[col].between(0, 1)
        if invalid_mask.any():
            invalid_indices = df.index[invalid_mask].tolist()
            invalid_values = df.loc[invalid_mask, col].tolist()
            logger.warning(f"Found {len(invalid_indices)} invalid values in {col}:")
            for idx, val in zip(invalid_indices, invalid_values):
                logger.warning(f"  Row {idx}: {val}")
        
        # Fix values
        df.loc[df[col] > 1, col] = 1.0
        df.loc[df[col] < 0, col] = 0.0
        df.loc[df[col].isna(), col] = 0.0
        
        # Round to 3 decimal places
        df[col] = df[col].round(3)
    
    return df

def clean_text(text: str) -> str:
    """Clean text fields"""
    if pd.isna(text):
        return ""
    return str(text).strip()

def transform_file(input_file: Path) -> Path:
    """Transform the classifier output file into importer format"""
    logger.info(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Validate and fix scores
    df = validate_and_fix_scores(df)
    
    # Clean text fields
    text_columns = ['use_case_name', 'agency', 'category_name', 'matched_keywords', 'explanation', 'error']
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    
    # Ensure abbreviation is present and clean
    if 'abbreviation' not in df.columns:
        logger.warning("Abbreviation column missing - will be derived from agency names")
        # Extract abbreviation from agency name if in parentheses
        df['abbreviation'] = df['agency'].str.extract(r'\((.*?)\)', expand=False)
        # For rows without parentheses, use first letters of words
        mask = df['abbreviation'].isna()
        df.loc[mask, 'abbreviation'] = df.loc[mask, 'agency'].apply(
            lambda x: ''.join(word[0].upper() for word in x.split() if word)
        )
    else:
        df['abbreviation'] = df['abbreviation'].apply(clean_text)
    
    # Select and rename columns to match importer expectations
    required_columns = [
        'use_case_name', 'agency', 'abbreviation', 'category_name',
        'keyword_score', 'semantic_score', 'llm_score', 'final_score',
        'match_method', 'relationship_type', 'confidence',
        'matched_keywords', 'explanation', 'error'
    ]
    
    # Verify all required columns exist
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create output file with _fixed suffix
    output_file = input_file.parent / f"{input_file.stem}_fixed{input_file.suffix}"
    
    # Select only required columns and save
    df[required_columns].to_csv(output_file, index=False)
    logger.info(f"Transformed file saved to: {output_file}")
    
    return output_file

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python transform_classifier_output.py <input_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        output_file = transform_file(input_file)
        print(f"Successfully transformed file. Output saved to: {output_file}")
    except Exception as e:
        print(f"Error transforming file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 