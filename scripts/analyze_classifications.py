"""
Analyze AI technology classification distributions from CSV export.

Usage:
    python scripts/analyze_classifications.py <csv_file>
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import contextlib

def analyze_classifications(csv_path: str):
    """Analyze classification distributions from CSV."""
    # Create a string buffer to capture output
    output = StringIO()
    
    # Redirect stdout to our string buffer
    with contextlib.redirect_stdout(output):
        print(f"\nAnalyzing classifications from: {csv_path}")
        print("-" * 80)
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        total_use_cases = df['use_case_name'].nunique()
        total_classifications = len(df)
        avg_classifications = total_classifications / total_use_cases
        
        print(f"Overview:")
        print(f"- Total unique use cases: {total_use_cases:,}")
        print(f"- Total classifications: {total_classifications:,}")
        print(f"- Average classifications per use case: {avg_classifications:.2f}")
        
        # Category distribution
        print(f"\nCategory Distribution:")
        cat_dist = df['category_name'].value_counts()
        for cat, count in cat_dist.items():
            pct = (count/total_classifications)*100
            print(f"- {cat}: {count:,} ({pct:.1f}%)")
        
        # Confidence score ranges
        print(f"\nConfidence Score Ranges:")
        ranges = [
            (0.90, 1.00, "Primary (0.90-1.00)"),
            (0.80, 0.89, "Supporting (0.80-0.89)"),
            (0.70, 0.79, "Related (0.70-0.79)"),
            (0.00, 0.69, "Below threshold (<0.70)")
        ]
        
        for low, high, label in ranges:
            count = len(df[(df['confidence_score'] >= low) & (df['confidence_score'] <= high)])
            pct = (count/total_classifications)*100
            print(f"- {label}: {count:,} ({pct:.1f}%)")
        
        # Agency distribution
        print(f"\nTop 10 Agencies by Number of Classifications:")
        agency_dist = df['agency_name'].value_counts().head(10)
        for agency, count in agency_dist.items():
            pct = (count/total_classifications)*100
            print(f"- {agency}: {count:,} ({pct:.1f}%)")
        
        # Average confidence by category
        print(f"\nAverage Confidence Score by Category:")
        avg_conf = df.groupby('category_name')['confidence_score'].mean().sort_values(ascending=False)
        for cat, score in avg_conf.items():
            print(f"- {cat}: {score:.3f}")
    
    # Get the output as string
    output_str = output.getvalue()
    
    # Print to console
    print(output_str)
    
    # Save to file
    csv_path = Path(csv_path)
    output_path = csv_path.parent / f"{csv_path.stem}_analysis.txt"
    with open(output_path, 'w') as f:
        f.write(output_str)
    
    print(f"\nAnalysis saved to: {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_classifications.py <csv_file>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
        
    try:
        analyze_classifications(csv_path)
    except Exception as e:
        print(f"Error analyzing classifications: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 