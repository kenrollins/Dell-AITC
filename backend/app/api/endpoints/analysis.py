from fastapi import APIRouter
from typing import List, Dict, Any
from ...models.analysis import AnalysisRequest, AnalysisMethod
from ...dependencies import get_classifier, get_neo4j_classifier

router = APIRouter()

@router.post("/analyze")
async def analyze_use_cases(
    method: AnalysisMethod = AnalysisMethod.ALL,
    dry_run: bool = False,
    limit: int = None,
    min_confidence: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Analyze use cases using the specified method
    """
    request = AnalysisRequest(method, dry_run, limit, min_confidence)
    classifier = get_classifier()
    neo4j_classifier = get_neo4j_classifier()
    
    # Get unclassified use cases
    use_cases = neo4j_classifier.get_unclassified_use_cases(limit=request.limit)
    
    results = []
    for use_case in use_cases:
        result = classifier.classify_use_case(use_case, method=request.method)
        
        if result['confidence'] >= request.min_confidence:
            results.append(result)
            
            if not request.dry_run and result['relationship_type'] != 'NO_MATCH':
                neo4j_classifier.save_classification(
                    use_case_id=result['use_case_id'],
                    category_name=result['category_name'],
                    confidence=result['confidence'],
                    match_method=result['match_method']
                )
    
    return results 