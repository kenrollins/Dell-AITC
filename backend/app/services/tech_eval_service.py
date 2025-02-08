import csv
import logging
from datetime import datetime

def save_results_to_csv(results, timestamp):
    """Save evaluation results to CSV files."""
    detailed_results = []
    neo4j_preview = []
    
    for result in results:
        # Keep original confidence value for detailed results
        detailed_results.append(result)
        
        # Format confidence as percentage only for preview
        confidence_pct = f"{result['confidence'] * 100:.1f}%"
        
        # Generate method-specific match details
        match_details = []
        if result['match_method'] == 'keyword':
            match_details.append(f"Keyword match ({result['keyword_score']:.3f})")
            if result['matched_keywords']:
                match_details.append(f"Matched terms: {', '.join(result['matched_keywords'])}")
        
        if result['semantic_score'] > 0:
            match_details.append(f"Semantic match ({result['semantic_score']:.3f})")
            
        if result['match_method'] == 'llm':
            match_details.append(f"LLM match ({result['llm_score']:.3f})")
            if result['llm_explanation']:
                match_details.append(f"LLM says: {result['llm_explanation']}")
                
        match_details_str = " | ".join(match_details)
        
        # Neo4j preview (simplified format)
        if result['relationship_type'] != 'NO_MATCH':
            neo4j_preview.append({
                'use_case_id': result['use_case_id'],
                'use_case_name': result['use_case_name'],
                'category_name': result['category_name'],
                'relationship_type': result['relationship_type'],
                'confidence': confidence_pct,
                'match_method': result['match_method'],
                'match_details': match_details_str
            })
    
    # Save detailed results
    detailed_file = f"data/output/results/ai_tech_classification_neo4j_{timestamp}.csv"
    with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
        writer.writeheader()
        writer.writerows(detailed_results)
    
    # Save Neo4j preview
    preview_file = f"data/output/results/ai_tech_classification_preview_{timestamp}.csv"
    with open(preview_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=neo4j_preview[0].keys())
        writer.writeheader()
        writer.writerows(neo4j_preview)
    
    logging.info(f"Saved detailed results to {detailed_file}")
    logging.info(f"Saved Neo4j preview to {preview_file}")
    
    return detailed_file, preview_file 