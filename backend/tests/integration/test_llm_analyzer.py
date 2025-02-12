"""
Integration tests for LLM Analyzer service.

Tests verify:
1. Match verification functionality
2. No-match analysis
3. Improvement suggestions
4. Integration with keyword and semantic matching

Usage:
    pytest test_llm_analyzer.py -v -n 1  # Test with 1 use case
    pytest test_llm_analyzer.py -v -n 5  # Test with 5 use cases
"""

import pytest
import pytest_asyncio
import json
import logging
from typing import Dict, List
from neo4j import AsyncGraphDatabase
from backend.app.services.llm_analyzer import LLMAnalyzer
from backend.app.services.classifier import Classifier
from backend.app.models.analysis import AnalysisMethod
from backend.app.config import get_settings
import respx
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of use cases to test
NUM_TEST_CASES = 1  # Change this to test more use cases

def load_schema_documentation() -> Dict:
    """Load Neo4j schema documentation for context."""
    schema_file = Path('docs/neo4j/neo4j_schema_documentation.md')
    if not schema_file.exists():
        logger.warning("Schema documentation not found!")
        return {}
        
    with open(schema_file, 'r') as f:
        content = f.read()
        
    # Extract relevant sections
    sections = {
        'nodes': {},
        'relationships': {}
    }
    
    # Parse node types
    node_section = content.split('## Node Types')[1].split('## Relationships')[0]
    for node_type in node_section.split('###')[1:]:
        if not node_type.strip():
            continue
        name = node_type.split('\n')[0].strip()
        sections['nodes'][name] = node_type
        
    # Parse relationships
    rel_section = content.split('## Relationships')[1].split('## Indexes')[0]
    for rel_type in rel_section.split('###')[1:]:
        if not rel_type.strip():
            continue
        name = rel_type.split('\n')[0].strip()
        sections['relationships'][name] = rel_type
        
    return sections

@pytest_asyncio.fixture
async def schema_context():
    """Load schema context for testing."""
    return load_schema_documentation()

@pytest_asyncio.fixture
async def mock_openai():
    """Mock OpenAI API responses."""
    with respx.mock(assert_all_called=False) as respx_mock:
        # Mock verify_match response with complete v2.2 schema properties
        verify_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "match_type": "PRIMARY",
                        "match_rank": 1,
                        "confidence": 0.92,
                        "analysis_method": "LLM",
                        "analysis_version": "v2.2",
                        "keyword_score": 0.85,
                        "semantic_score": 0.88,
                        "llm_score": 0.92,
                        "field_match_scores": {
                            "technical_alignment": 0.87,
                            "business_alignment": 0.93,
                            "implementation_fit": 0.85,
                            "capability_coverage": 0.90,
                            "maturity_alignment": 0.88
                        },
                        "term_match_details": {
                            "matched_keywords": ["NLP", "text analysis", "language processing"],
                            "matched_capabilities": ["text understanding", "document analysis"],
                            "matched_business_terms": ["efficiency", "automation"],
                            "context_matches": ["document processing workflow", "text extraction"]
                        },
                        "matched_keywords": ["NLP", "text analysis", "language processing"],
                        "llm_verification": True,
                        "llm_confidence": 0.92,
                        "llm_reasoning": "Strong alignment with NLP capabilities and use case requirements",
                        "llm_suggestions": [
                            "Add specific NLP libraries used",
                            "Detail accuracy metrics"
                        ],
                        "improvement_notes": [
                            "Add more technical implementation details",
                            "Clarify integration points",
                            "Specify performance metrics"
                        ],
                        "false_positive": False,
                        "manual_override": False,
                        "review_status": "PENDING"
                    })
                }
            }]
        }
        
        # Mock no_match response with complete v2.2 schema properties
        no_match_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "reason": "Insufficient technical details and unclear AI implementation",
                        "confidence": 0.85,
                        "llm_analysis": {
                            "primary_focus": "Document processing",
                            "tech_stack": ["OCR", "NLP"],
                            "missing_elements": ["technical details", "implementation approach"],
                            "potential_categories": ["Document Intelligence", "Text Analytics"]
                        },
                        "suggested_keywords": ["OCR", "document analysis", "text extraction"],
                        "improvement_suggestions": {
                            "category_updates": ["Add document processing focus"],
                            "keyword_additions": ["OCR", "layout analysis"],
                            "new_categories": ["Document Intelligence"]
                        },
                        "status": "NEW",
                        "review_notes": "Requires more technical implementation details"
                    })
                }
            }]
        }
        
        # Mock improvement suggestions response
        improvement_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "definition_updates": {
                            "current": "Existing definition...",
                            "suggested": "Updated definition with clearer scope...",
                            "reasoning": "Better reflects current use cases"
                        },
                        "keyword_updates": {
                            "add": ["neural networks", "deep learning"],
                            "remove": ["outdated term"],
                            "reasoning": "Align with current technology"
                        },
                        "capability_updates": {
                            "add": ["real-time processing", "batch analysis"],
                            "remove": [],
                            "reasoning": "Cover observed use patterns"
                        },
                        "business_term_updates": {
                            "add": ["operational efficiency", "cost reduction"],
                            "remove": ["legacy terms"],
                            "reasoning": "Match business objectives"
                        },
                        "match_criteria_updates": {
                            "threshold_adjustments": {"confidence": 0.4},
                            "scoring_weights": {"technical": 0.6, "business": 0.4},
                            "reasoning": "Optimize classification accuracy"
                        }
                    })
                }
            }]
        }
        
        # Register mock responses
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=verify_response),
                httpx.Response(200, json=no_match_response),
                httpx.Response(200, json=improvement_response)
            ]
        )
        
        yield respx_mock

@pytest_asyncio.fixture
async def neo4j_driver():
    """Create Neo4j driver instance."""
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    yield driver
    await driver.close()

@pytest_asyncio.fixture
async def classifier():
    """Create classifier instance for testing."""
    return Classifier(dry_run=True)

@pytest_asyncio.fixture
async def sample_use_cases(neo4j_driver):
    """Fetch sample use cases from Neo4j with all fields needed for LLM analysis."""
    async with neo4j_driver.session() as session:
        # Get random use cases that don't have classifications yet
        query = """
        MATCH (u:UseCase)
        WHERE NOT EXISTS((u)-[:USES_TECHNOLOGY]->(:AICategory))
        WITH u, rand() as r
        ORDER BY r
        LIMIT $limit
        RETURN {
            id: u.id,
            name: u.name,
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            topic_area: u.topic_area,
            stage: u.stage,
            impact_type: u.impact_type,
            dev_method: u.dev_method,
            system_name: u.system_name
        } as use_case
        """
        result = await session.run(query, limit=NUM_TEST_CASES)
        use_cases = [record["use_case"] async for record in result]
        
        if not use_cases:
            # If no unclassified cases found, get any use cases
            query = """
            MATCH (u:UseCase)
            WITH u, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN {
                id: u.id,
                name: u.name,
                description: u.description,
                purpose_benefits: u.purpose_benefits,
                outputs: u.outputs,
                topic_area: u.topic_area,
                stage: u.stage,
                impact_type: u.impact_type,
                dev_method: u.dev_method,
                system_name: u.system_name
            } as use_case
            """
            result = await session.run(query, limit=NUM_TEST_CASES)
            use_cases = [record["use_case"] async for record in result]
            
        logger.info(f"Retrieved {len(use_cases)} use cases from Neo4j:")
        for uc in use_cases:
            logger.info(f"  - {uc['id']}: {uc['name']}")
            
        return use_cases

@pytest_asyncio.fixture
async def ai_categories(neo4j_driver):
    """Fetch AI categories with complete definitions from Neo4j."""
    async with neo4j_driver.session() as session:
        query = """
        MATCH (c:AICategory)
        WHERE c.status = 'active'
        OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
        OPTIONAL MATCH (c)-[:HAS_CAPABILITY]->(cap:Capability)
        WITH c, 
             collect(DISTINCT k.name) as keywords,
             collect(DISTINCT cap.name) as capabilities
        RETURN {
            id: c.id,
            name: c.name,
            definition: c.definition,
            status: c.status,
            maturity_level: c.maturity_level,
            zone: c.zone,
            keywords: keywords,
            capabilities: capabilities,
            business_language: c.business_language
        } as category
        """
        result = await session.run(query)
        categories = [record["category"] async for record in result]
        
        logger.info(f"Retrieved {len(categories)} AI categories from Neo4j:")
        for cat in categories:
            logger.info(f"  - {cat['name']}")
            logger.info(f"    Keywords: {len(cat['keywords'])}")
            logger.info(f"    Capabilities: {len(cat['capabilities'])}")
            
        return categories

@pytest_asyncio.fixture
async def llm_analyzer(mock_openai):
    """Create LLM analyzer instance for testing."""
    analyzer = LLMAnalyzer()
    await analyzer.initialize()
    yield analyzer
    await analyzer.cleanup()

@pytest.mark.asyncio
async def test_verify_matches(llm_analyzer, sample_use_cases, ai_categories):
    """Test LLM verification of potential matches."""
    logger.info(f"Testing with {len(sample_use_cases)} use cases")
    
    for use_case in sample_use_cases:
        logger.info(f"\nAnalyzing use case: {use_case['id']} - {use_case['name']}")
        
        # Format use case text for LLM analysis
        use_case_text = f"""
        Use Case Details:
        ID: {use_case['id']}
        Name: {use_case['name']}
        Description: {use_case.get('description', 'N/A')}
        Purpose & Benefits: {use_case.get('purpose_benefits', 'N/A')}
        Outputs: {use_case.get('outputs', 'N/A')}
        Topic Area: {use_case.get('topic_area', 'N/A')}
        Stage: {use_case.get('stage', 'N/A')}
        Impact Type: {use_case.get('impact_type', 'N/A')}
        Development Method: {use_case.get('dev_method', 'N/A')}
        System Name: {use_case.get('system_name', 'N/A')}
        """
        
        # Test against each active category
        for category in ai_categories:
            logger.info(f"Testing against category: {category['name']}")
            
            try:
                # Call LLM to verify the match
                result = await llm_analyzer.verify_match(
                    use_case_text=use_case_text,
                    category_name=category['name'],
                    match_type="PRIMARY",  # Initial match type for testing
                    confidence=0.5  # Initial confidence for testing
                )
                
                # Validate complete v2.2 schema properties
                assert isinstance(result, dict), "Result should be a dictionary"
                
                # Required fields
                assert isinstance(result.get("match_type"), str), "Match type should be string"
                assert result["match_type"] in ["PRIMARY", "SUPPORTING", "RELATED", "NONE"], "Invalid match type"
                assert isinstance(result.get("confidence"), float), "Confidence should be float"
                assert 0.0 <= result["confidence"] <= 1.0, "Confidence should be between 0 and 1"
                assert isinstance(result.get("analysis_method"), str), "Analysis method should be string"
                assert result["analysis_method"] in ["KEYWORD", "SEMANTIC", "LLM", "ENSEMBLE"], "Invalid analysis method"
                
                # Optional fields with type validation
                assert isinstance(result.get("match_rank", 0), int), "Match rank should be integer"
                assert isinstance(result.get("analysis_version", ""), str), "Analysis version should be string"
                assert isinstance(result.get("keyword_score", 0.0), float), "Keyword score should be float"
                assert isinstance(result.get("semantic_score", 0.0), float), "Semantic score should be float"
                assert isinstance(result.get("llm_score", 0.0), float), "LLM score should be float"
                
                # Complex field validations
                assert isinstance(result.get("field_match_scores", {}), dict), "Field match scores should be dict"
                if "field_match_scores" in result:
                    for score_name, score in result["field_match_scores"].items():
                        assert isinstance(score, float), f"Field score {score_name} should be float"
                        assert 0.0 <= score <= 1.0, f"Field score {score_name} should be between 0 and 1"
                
                assert isinstance(result.get("term_match_details", {}), dict), "Term match details should be dict"
                if "term_match_details" in result:
                    for term_list in result["term_match_details"].values():
                        assert isinstance(term_list, list), "Term matches should be lists"
                        assert all(isinstance(term, str) for term in term_list), "Terms should be strings"
                
                assert isinstance(result.get("matched_keywords", []), list), "Matched keywords should be list"
                assert all(isinstance(kw, str) for kw in result.get("matched_keywords", [])), "Keywords should be strings"
                
                # LLM-specific fields
                assert isinstance(result.get("llm_verification"), bool), "LLM verification should be boolean"
                assert isinstance(result.get("llm_confidence", 0.0), float), "LLM confidence should be float"
                assert isinstance(result.get("llm_reasoning", ""), str), "LLM reasoning should be string"
                assert isinstance(result.get("llm_suggestions", []), list), "LLM suggestions should be list"
                assert isinstance(result.get("improvement_notes", []), list), "Improvement notes should be list"
                
                # Status fields
                assert isinstance(result.get("false_positive"), bool), "False positive should be boolean"
                assert isinstance(result.get("manual_override"), bool), "Manual override should be boolean"
                assert result.get("review_status") in ["PENDING", "REVIEWED", "VERIFIED"], "Invalid review status"
                
                # Log results
                if result["llm_verification"]:
                    logger.info(f"✓ Match found: {category['name']} ({result['match_type']})")
                    logger.info(f"  Confidence: {result['confidence']:.2f}")
                    logger.info(f"  Match Rank: {result.get('match_rank', 'N/A')}")
                    logger.info(f"  Analysis Method: {result['analysis_method']}")
                    logger.info(f"  Reasoning: {result['llm_reasoning']}")
                    logger.info(f"  Field Scores: {json.dumps(result.get('field_match_scores', {}), indent=2)}")
                    logger.info(f"  Term Matches: {json.dumps(result.get('term_match_details', {}), indent=2)}")
                    logger.info(f"  Review Status: {result['review_status']}")
                else:
                    logger.info(f"✗ No match with {category['name']}")
                    logger.info(f"  Reason: {result['llm_reasoning']}")
                
            except Exception as e:
                logger.error(f"Error analyzing category {category['name']}: {str(e)}")
                raise

@pytest.mark.asyncio
async def test_analyze_no_matches(llm_analyzer, sample_use_cases, ai_categories):
    """Test LLM analysis of use cases with no clear matches."""
    logger.info("\nTesting analysis of unmatched use cases")
    
    for use_case in sample_use_cases:
        logger.info(f"\nAnalyzing use case: {use_case['id']} - {use_case['name']}")
        
        # Format use case text with all available fields
        use_case_text = f"""
        Use Case Details:
        ID: {use_case['id']}
        Name: {use_case['name']}
        Description: {use_case.get('description', 'N/A')}
        Purpose & Benefits: {use_case.get('purpose_benefits', 'N/A')}
        Outputs: {use_case.get('outputs', 'N/A')}
        Topic Area: {use_case.get('topic_area', 'N/A')}
        Stage: {use_case.get('stage', 'N/A')}
        Impact Type: {use_case.get('impact_type', 'N/A')}
        Development Method: {use_case.get('dev_method', 'N/A')}
        System Name: {use_case.get('system_name', 'N/A')}
        """
        
        # Add context about available categories
        category_context = "\nAvailable Technology Categories:\n"
        for cat in ai_categories:
            category_context += f"""
            - {cat['name']}:
              Definition: {cat.get('definition', 'N/A')}
              Keywords: {', '.join(cat.get('keywords', []))}
              Capabilities: {', '.join(cat.get('capabilities', []))}
            """
        
        # Analyze with full context
        result = await llm_analyzer.analyze_no_match(use_case_text + category_context)
        
        # Validate complete v2.2 schema properties for NoMatchAnalysis
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Required fields
        assert isinstance(result.get("reason"), str), "Reason should be string"
        assert len(result["reason"]) > 0, "Reason should not be empty"
        
        # Optional fields with type validation
        assert isinstance(result.get("confidence", 0.0), float), "Confidence should be float"
        if "confidence" in result:
            assert 0.0 <= result["confidence"] <= 1.0, "Confidence should be between 0 and 1"
        
        # Complex field validations
        assert isinstance(result.get("llm_analysis", {}), dict), "LLM analysis should be dict"
        if "llm_analysis" in result:
            assert isinstance(result["llm_analysis"].get("primary_focus", ""), str), "Primary focus should be string"
            assert isinstance(result["llm_analysis"].get("tech_stack", []), list), "Tech stack should be list"
            assert isinstance(result["llm_analysis"].get("missing_elements", []), list), "Missing elements should be list"
            assert isinstance(result["llm_analysis"].get("potential_categories", []), list), "Potential categories should be list"
        
        assert isinstance(result.get("suggested_keywords", []), list), "Suggested keywords should be list"
        if "suggested_keywords" in result:
            assert all(isinstance(kw, str) for kw in result["suggested_keywords"]), "Keywords should be strings"
        
        assert isinstance(result.get("improvement_suggestions", {}), dict), "Improvement suggestions should be dict"
        if "improvement_suggestions" in result:
            assert isinstance(result["improvement_suggestions"].get("category_updates", []), list), "Category updates should be list"
            assert isinstance(result["improvement_suggestions"].get("keyword_additions", []), list), "Keyword additions should be list"
            assert isinstance(result["improvement_suggestions"].get("new_categories", []), list), "New categories should be list"
        
        # Status fields
        assert result.get("status", "NEW") in ["NEW", "REVIEWED", "ACTIONED"], "Invalid status"
        assert isinstance(result.get("review_notes", ""), str), "Review notes should be string"
        
        # Log detailed results
        logger.info(f"\nAnalysis Results for {use_case['name']}:")
        logger.info(f"Reason: {result['reason']}")
        logger.info(f"Confidence: {result.get('confidence', 0.0):.2f}")
        
        if result.get("llm_analysis"):
            logger.info("LLM Analysis:")
            logger.info(f"  Primary Focus: {result['llm_analysis'].get('primary_focus', 'N/A')}")
            logger.info(f"  Tech Stack: {', '.join(result['llm_analysis'].get('tech_stack', []))}")
            logger.info(f"  Missing Elements: {', '.join(result['llm_analysis'].get('missing_elements', []))}")
            logger.info(f"  Potential Categories: {', '.join(result['llm_analysis'].get('potential_categories', []))}")
        
        if result.get("suggested_keywords"):
            logger.info(f"Suggested Keywords: {', '.join(result['suggested_keywords'])}")
            
        if result.get("improvement_suggestions"):
            logger.info("Improvement Suggestions:")
            for key, values in result["improvement_suggestions"].items():
                logger.info(f"  {key}: {', '.join(values)}")
        
        logger.info(f"Status: {result.get('status', 'NEW')}")
        if result.get("review_notes"):
            logger.info(f"Review Notes: {result['review_notes']}")

@pytest.mark.asyncio
async def test_suggest_improvements(llm_analyzer, ai_categories):
    """Test improvement suggestions for categories."""
    if not ai_categories:
        pytest.skip("No AI categories available for testing")
        
    category = ai_categories[0]
    
    # Sample match data
    recent_matches = [
        {
            "use_case_id": "UC001",
            "confidence": 0.82,
            "match_type": "PRIMARY",
            "matched_keywords": ["NLP", "text analysis"],
            "field_scores": {"technical": 0.85, "business": 0.78}
        }
    ]
    
    recent_failures = [
        {
            "use_case_id": "UC002",
            "confidence": 0.28,
            "reason": "Insufficient technical detail",
            "missing_elements": ["implementation approach", "algorithms"]
        }
    ]
    
    result = await llm_analyzer.suggest_improvements(
        category_name=category["name"],
        recent_matches=recent_matches,
        recent_failures=recent_failures
    )
    
    # Validate improvement suggestion structure
    assert isinstance(result, dict)
    
    assert isinstance(result.get("definition_updates"), dict)
    assert "current" in result["definition_updates"]
    assert "suggested" in result["definition_updates"]
    assert "reasoning" in result["definition_updates"]
    
    assert isinstance(result.get("keyword_updates"), dict)
    assert isinstance(result["keyword_updates"].get("add"), list)
    assert isinstance(result["keyword_updates"].get("remove"), list)
    assert isinstance(result["keyword_updates"].get("reasoning"), str)
    
    assert isinstance(result.get("capability_updates"), dict)
    assert isinstance(result["capability_updates"].get("add"), list)
    assert isinstance(result["capability_updates"].get("remove"), list)
    assert isinstance(result["capability_updates"].get("reasoning"), str)
    
    assert isinstance(result.get("business_term_updates"), dict)
    assert isinstance(result["business_term_updates"].get("add"), list)
    assert isinstance(result["business_term_updates"].get("remove"), list)
    assert isinstance(result["business_term_updates"].get("reasoning"), str)
    
    assert isinstance(result.get("match_criteria_updates"), dict)
    assert isinstance(result["match_criteria_updates"].get("threshold_adjustments"), dict)
    assert isinstance(result["match_criteria_updates"].get("scoring_weights"), dict)
    assert isinstance(result["match_criteria_updates"].get("reasoning"), str)
    
    logger.info(f"Improvement suggestions for {category['name']}: {json.dumps(result, indent=2)}")

@pytest.mark.asyncio
async def test_error_handling(llm_analyzer):
    """Test error handling for empty use case text."""
    result = await llm_analyzer.analyze_no_match("")
    assert result["reason_category"] == "UNCLEAR_DESC"
    assert result["confidence"] == 0.0
    assert result["reason"].startswith("Error")
    assert not result["suggested_categories"]

@pytest.mark.asyncio
async def test_error_handling_extended(llm_analyzer):
    """Test extended error handling scenarios."""
    # Mock OpenAI client to simulate an error response
    mock_openai = mocker.patch('openai.OpenAI')
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = Exception("Invalid response format")

    # Test with invalid input
    result = await llm_analyzer.analyze_no_match("test input")
    assert result["reason_category"] == "PROCESSING_ERROR"
    assert result["confidence"] == 0.0
    assert "Error processing request" in result["reason"]
    assert not result["suggested_categories"]

@pytest.mark.asyncio
async def test_load_handling(llm_analyzer, sample_use_cases, ai_categories):
    """Test system performance with larger batches of use cases."""
    if not sample_use_cases or len(sample_use_cases) < 2:
        pytest.skip("Not enough sample use cases for load testing")

    # Process multiple use cases concurrently
    import asyncio
    tasks = []
    for use_case in sample_use_cases[:5]:  # Test with up to 5 concurrent requests
        use_case_text = f"""
        Use Case Details:
        ID: {use_case['id']}
        Name: {use_case['name']}
        Description: {use_case.get('description', 'N/A')}
        Purpose & Benefits: {use_case.get('purpose_benefits', 'N/A')}
        """
        
        tasks.append(llm_analyzer.analyze_no_match(use_case_text))
    
    # Execute concurrent requests
    results = await asyncio.gather(*tasks)
    
    # Verify all results
    for result in results:
        assert isinstance(result, dict)
        assert "reason" in result
        assert "reason_category" in result
        assert "confidence" in result
        assert isinstance(result["llm_analysis"], dict)
        assert isinstance(result["suggested_keywords"], list)
        assert isinstance(result["improvement_suggestions"], dict)

@pytest.mark.asyncio
async def test_category_validation(llm_analyzer):
    """Test category validation and error handling."""
    
    # Test with non-existent category
    with pytest.raises(ValueError, match="Unknown category"):
        await llm_analyzer.verify_match(
            use_case_text="Sample text",
            category_name="NonexistentCategory",
            match_type="PRIMARY",
            confidence=0.5
        )
    
    # Test with invalid match type
    if llm_analyzer.category_definitions:
        category_name = next(iter(llm_analyzer.category_definitions.keys()))
        with pytest.raises(ValueError, match="Invalid match type"):
            await llm_analyzer.verify_match(
                use_case_text="Sample text",
                category_name=category_name,
                match_type="INVALID",
                confidence=0.5
            )

    # Test suggest_improvements with invalid category
    with pytest.raises(ValueError, match="Unknown category"):
        await llm_analyzer.suggest_improvements(
            category_name="NonexistentCategory",
            recent_matches=[],
            recent_failures=[]
        )

@pytest.mark.asyncio
async def test_neo4j_error_handling(llm_analyzer):
    """Test handling of Neo4j connection issues."""
    
    # Simulate Neo4j connection failure
    if llm_analyzer.neo4j_driver:
        await llm_analyzer.neo4j_driver.close()
        llm_analyzer.neo4j_driver = None
    
    # Attempt to reload category definitions
    try:
        await llm_analyzer._load_category_definitions()
        assert len(llm_analyzer.category_definitions) == 0
    except Exception as e:
        assert "Neo4j" in str(e) 