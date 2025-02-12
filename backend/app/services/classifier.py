import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from ..models.analysis import AnalysisMethod
from ..models.ai_category import get_categories
from ..utils.text_processor import TextProcessor
from ..services.llm_analyzer import LLMAnalyzer
from neo4j import GraphDatabase
from app.config import Settings, get_settings
from neo4j import AsyncGraphDatabase
import httpx
from openai import AsyncOpenAI, OpenAI
import asyncio
from ..models.match_type import MatchType

logger = logging.getLogger(__name__)

class Classifier:
    def __init__(self, dry_run: bool = False):
        """Initialize classifier with configuration."""
        self.settings = get_settings()
        self.dry_run = dry_run
        self.driver = None
        self.text_processor = None
        self.llm_analyzer = None
        self.categories = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connections and load dependencies."""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
            
        if not self.text_processor:
            self.text_processor = TextProcessor()
            
        if not self.llm_analyzer:
            self.llm_analyzer = LLMAnalyzer()
            await self.llm_analyzer.initialize()
            
        if not self.categories:
            self.categories = await get_categories()
            
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=30.0
        )
            
        logging.info(f"Classifier initialized with {len(self.categories)} categories")
        
    async def close(self):
        """Cleanup connections."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            
        if self.llm_analyzer:
            await self.llm_analyzer.cleanup()
            self.llm_analyzer = None
            
    def __del__(self):
        """Ensure connections are closed."""
        if self.driver:
            try:
                asyncio.create_task(self.close())
                logging.debug("Classifier cleanup initiated")
            except Exception as e:
                logging.error(f"Error during classifier cleanup: {str(e)}")

    async def get_use_case_by_id(self, use_case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a use case by ID from Neo4j.
        
        Args:
            use_case_id: The ID of the use case to fetch
            
        Returns:
            Dictionary containing use case data or None if not found
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (u:UseCase {id: $use_case_id})
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
                result = await session.run(query, use_case_id=use_case_id)
                record = await result.single()
                
                if not record:
                    self.logger.error(f"Use case {use_case_id} not found")
                    return None
                    
                return record["use_case"]
                
        except Exception as e:
            self.logger.error(f"Error fetching use case {use_case_id}: {str(e)}")
            return None

    async def classify_use_case(self, use_case: Dict[str, Any], method: AnalysisMethod = AnalysisMethod.ALL, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Classify a use case using the specified method(s).
        Primary classification uses keyword and semantic matching.
        LLM analysis is used as a fallback for low confidence or validation.
        """
        results = {}

        # Step 1: Perform keyword analysis
        keyword_results = self._perform_keyword_analysis(use_case)
        best_match_category = keyword_results.pop('_best_match_category', None)
        category_name = keyword_results.pop('category_name', None)
        match_details = keyword_results.pop('match_details', {})

        # Initialize results with base structure
        results = {
            'use_case_id': use_case['id'],
            'use_case_name': use_case['name'],
            'category_name': category_name,
            'match_type': MatchType.NO_MATCH.value,
            'confidence': 0.0,
            'keyword_score': keyword_results['keyword_score'],
            'semantic_score': 0.0,
            'llm_score': 0.0,
            'matched_keywords': keyword_results['matched_keywords'],
            'match_method': '',
            'llm_explanation': '',
            'match_details': match_details
        }

        # Step 2: Perform semantic analysis if needed
        if method in [AnalysisMethod.SEMANTIC, AnalysisMethod.ALL]:
            semantic_results = self._perform_semantic_analysis(use_case)
            
            # Find best semantic match
            best_semantic = max(semantic_results.items(), key=lambda x: x[1]['score'], default=(None, {'score': 0}))
            results['semantic_score'] = best_semantic[1]['score']
            
            # Update confidence based on combined scores
            if best_match_category:
                semantic_score = semantic_results.get(best_match_category, {}).get('score', 0)
                results['confidence'] = max(
                    results['keyword_score'],
                    semantic_score,
                    (results['keyword_score'] * 0.7 + semantic_score * 0.3)
                )

        # Step 3: Evaluate initial results
        best_match = None
        supporting_matches = []
        related_matches = []

        # Evaluate keyword and semantic scores
        if results['keyword_score'] >= 0.45 or results['semantic_score'] >= 0.45:
            # Strong match - Primary
            results['match_type'] = MatchType.PRIMARY.value
            results['confidence'] = max(results['keyword_score'], results['semantic_score'])
            results['match_method'] = 'ENSEMBLE'
            
        elif results['keyword_score'] >= 0.35 or results['semantic_score'] >= 0.40:
            # Medium confidence - Supporting
            results['match_type'] = MatchType.SUPPORTING.value
            results['confidence'] = max(results['keyword_score'], results['semantic_score']) * 0.85
            results['match_method'] = 'KEYWORD' if results['keyword_score'] > results['semantic_score'] else 'SEMANTIC'
            
        elif results['keyword_score'] >= 0.25 or results['semantic_score'] >= 0.30:
            # Lower confidence - Related
            results['match_type'] = MatchType.RELATED.value
            results['confidence'] = (results['keyword_score'] * 0.6 + results['semantic_score'] * 0.4)
            results['match_method'] = 'ENSEMBLE'
            
        else:
            # No clear match - Use LLM
            results['match_type'] = MatchType.NO_MATCH.value
            results['confidence'] = max(results['keyword_score'], results['semantic_score']) * 0.5
            results['match_method'] = 'INSUFFICIENT_CONFIDENCE'

        # Step 4: Use LLM as fallback for validation or low confidence cases
        if method in [AnalysisMethod.LLM, AnalysisMethod.ALL]:
            should_use_llm = (
                results['match_type'] == MatchType.NO_MATCH.value or  # No strong matches found
                results['confidence'] < 0.45  # Primary match needs validation
            )

            if should_use_llm:
                llm_results = await self._perform_llm_analysis(use_case)
                
                # Update results with LLM analysis
                if llm_results['llm_score'] > 0.4:  # LLM found a good match
                    if llm_results['llm_score'] > results['confidence']:
                        # LLM found a better match
                        results.update(llm_results)
                        results['match_type'] = MatchType.PRIMARY.value
                        results['match_method'] = 'LLM'
                    elif llm_results['llm_score'] >= 0.35:
                        # LLM found a supporting match
                        supporting_matches.append(llm_results)
                    else:
                        # LLM found a related match
                        related_matches.append(llm_results)

        # Step 5: Add supporting and related matches if found
        if supporting_matches:
            results['supporting_matches'] = supporting_matches[:2]  # Limit to top 2
            
        if related_matches:
            results['related_matches'] = related_matches[:3]  # Limit to top 3

        # Save results to database if requested
        if save_to_db and not self.dry_run:
            await self._save_to_neo4j(results)

        return results

    def classify_by_use_case_number(self, use_case_number: str, method: AnalysisMethod = AnalysisMethod.ALL, save_to_db: bool = True) -> Optional[Dict[str, Any]]:
        """
        Classify a specific use case by its number/ID
        
        Args:
            use_case_number: The use case ID to classify
            method: Analysis method to use
            save_to_db: Whether to save results to Neo4j
        """
        with self.driver.session() as session:
            # Fetch use case data
            query = """
            MATCH (u:UseCase {id: $use_case_id})
            RETURN {
                id: u.id,
                name: u.name,
                description: u.description,
                purpose_benefits: u.purpose_benefits,
                outputs: u.outputs
            } as use_case
            """
            result = session.run(query, use_case_id=use_case_number)
            record = result.single()
            
            if not record:
                self.logger.error(f"Use case {use_case_number} not found")
                return None
                
            use_case = record['use_case']
            # In dry run mode, we classify but don't save
            return self.classify_use_case(use_case, method, save_to_db=not self.dry_run)

    async def _save_to_neo4j(self, results: Dict[str, Any]) -> None:
        """Save classification results to Neo4j using v2.2 schema"""
        if self.dry_run:
            self.logger.info("Dry run mode - skipping Neo4j save")
            return
            
        async with self.driver.session() as session:
            # Create AIClassification node and relationships
            query = """
            MATCH (u:UseCase {id: $use_case_id})
            MATCH (c:AICategory {name: $category_name})
            
            // Create AIClassification node
            CREATE (cl:AIClassification {
                id: apoc.create.uuid(),
                match_type: $match_type,
                match_rank: $match_rank,
                confidence: $confidence,
                analysis_method: $analysis_method,
                analysis_version: 'v2.2',
                keyword_score: $keyword_score,
                semantic_score: $semantic_score,
                llm_score: $llm_score,
                field_match_scores: $field_match_scores,
                term_match_details: $term_match_details,
                matched_keywords: $matched_keywords,
                llm_verification: $llm_verification,
                llm_confidence: $llm_confidence,
                llm_reasoning: $llm_reasoning,
                llm_suggestions: $llm_suggestions,
                improvement_notes: $improvement_notes,
                false_positive: $false_positive,
                manual_override: $manual_override,
                review_status: $review_status,
                classified_at: datetime(),
                classified_by: $classified_by,
                last_updated: datetime()
            })
            
            // Create relationships
            CREATE (c)-[:CLASSIFIES]->(cl)
            CREATE (u)-[:CLASSIFIED_AS]->(cl)
            
            RETURN cl
            """
            
            # Determine match type and rank
            match_type = (
                'PRIMARY' if results['confidence'] >= 0.45
                else 'SUPPORTING' if results['confidence'] >= 0.35
                else 'RELATED'
            )
            
            await session.run(
                query,
                use_case_id=results['use_case_id'],
                category_name=results['category_name'],
                match_type=match_type,
                match_rank=1,  # Default to 1 for now
                confidence=results['confidence'],
                analysis_method=results['match_method'].upper(),
                keyword_score=results['keyword_score'],
                semantic_score=results['semantic_score'],
                llm_score=results['llm_score'],
                field_match_scores=results.get('field_match_scores', {}),
                term_match_details=results.get('term_match_details', {}),
                matched_keywords=results['matched_keywords'],
                llm_verification=results.get('llm_verification', False),
                llm_confidence=results.get('llm_confidence', 0.0),
                llm_reasoning=results.get('llm_reasoning', ''),
                llm_suggestions=results.get('llm_suggestions', []),
                improvement_notes=results.get('improvement_notes', []),
                false_positive=results.get('false_positive', False),
                manual_override=results.get('manual_override', False),
                review_status='PENDING',
                classified_by='system'  # Default to system for automated classification
            )
            
            # If no match, create NoMatchAnalysis node
            if not results['category_name'] or results['confidence'] < 0.3:
                no_match_query = """
                MATCH (u:UseCase {id: $use_case_id})
                
                // Create NoMatchAnalysis node
                CREATE (na:NoMatchAnalysis {
                    id: apoc.create.uuid(),
                    reason: $reason,
                    confidence: $confidence,
                    llm_analysis: $llm_analysis,
                    suggested_keywords: $suggested_keywords,
                    improvement_suggestions: $improvement_suggestions,
                    created_at: datetime(),
                    analyzed_by: 'system',
                    status: 'NEW',
                    review_notes: $review_notes
                })
                
                // Create relationship
                CREATE (u)-[:HAS_ANALYSIS]->(na)
                
                RETURN na
                """
                
                await session.run(
                    no_match_query,
                    use_case_id=results['use_case_id'],
                    reason=results.get('no_match_reason', 'Insufficient confidence'),
                    confidence=results['confidence'],
                    llm_analysis=results.get('llm_analysis', {}),
                    suggested_keywords=results.get('suggested_keywords', []),
                    improvement_suggestions=results.get('improvement_suggestions', []),
                    review_notes=results.get('review_notes', '')
                )
            
            self.logger.info(f"Saved classification results for use case {results['use_case_id']} to Neo4j")

    def _perform_keyword_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """Perform keyword-based analysis with improved confidence scoring"""
        # Combine relevant text fields for analysis, safely handling None values
        text_fields = {
            'name': str(use_case.get('name', '') or '').lower(),
            'description': str(use_case.get('description', '') or '').lower(),
            'purpose_benefits': str(use_case.get('purpose_benefits', '') or '').lower(),
            'outputs': str(use_case.get('outputs', '') or '').lower()
        }
        
        # Field weights (increased importance of name and purpose)
        field_weights = {
            'name': 4.0,        # Increased from 3.0
            'description': 0.8,  # Kept same
            'purpose_benefits': 2.5,  # Increased from 2.0
            'outputs': 1.5      # Increased from 1.2
        }
        
        best_match: Tuple[str, float, List[str], Dict[str, Any]] = ('', 0.0, [], {})
        
        # Check each category
        for category_name, category in self.categories.items():
            matched_keywords = []
            field_matches = {field: [] for field in text_fields.keys()}
            term_weights = {}
            
            # Check each field
            for field_name, field_text in text_fields.items():
                # Technical keywords (highest weight)
                for keyword in category.keywords:
                    if not keyword:  # Skip empty keywords
                        continue
                    keyword_lower = str(keyword).lower()
                    if keyword_lower in field_text:
                        matched_keywords.append(keyword)
                        field_matches[field_name].append(keyword)
                        # Boost weight for exact matches in name or purpose
                        if field_name in ['name', 'purpose_benefits'] and keyword_lower == field_text:
                            term_weights[keyword] = 2.0 * field_weights[field_name]
                        else:
                            term_weights[keyword] = 1.5 * field_weights[field_name]
                
                # Capability terms (medium weight)
                for capability in category.capabilities:
                    if not capability:  # Skip empty capabilities
                        continue
                    capability_lower = str(capability).lower()
                    if capability_lower in field_text:
                        matched_keywords.append(capability)
                        field_matches[field_name].append(capability)
                        term_weights[capability] = 1.2 * field_weights[field_name]
                
                # Business language terms (lower weight)
                for term in category.business_language:
                    if not term:  # Skip empty terms
                        continue
                    term_lower = str(term).lower()
                    if term_lower in field_text:
                        matched_keywords.append(term)
                        field_matches[field_name].append(term)
                        term_weights[term] = 0.9 * field_weights[field_name]
            
            # Calculate score if we have matches
            if matched_keywords:
                # Remove duplicates while preserving order
                matched_keywords = list(dict.fromkeys(matched_keywords))
                
                # Base score from term weights
                base_score = sum(term_weights[kw] for kw in matched_keywords) * 1.5
                
                # Calculate match density
                field_densities = {
                    field: len(matches) / max(len(field_text.split()) / 3, 1)
                    for field, matches in field_matches.items()
                    for field_text in [text_fields[field]]
                    if matches
                }
                avg_density = sum(field_densities.values()) / len(field_densities) if field_densities else 0
                
                # Calculate diversity score
                fields_with_matches = sum(1 for matches in field_matches.values() if matches)
                field_diversity = (fields_with_matches / len(field_matches)) * 2.0
                
                # Combine scores with adjusted weights
                final_score = (
                    base_score * 0.7 +
                    avg_density * 0.2 +
                    field_diversity * 0.1
                ) / (
                    max(len(category.keywords) * 0.3, 1)
                )
                
                # Add score boosts for strong matches
                if fields_with_matches >= 2:
                    final_score *= 1.3
                
                if len(matched_keywords) >= 3:
                    final_score *= 1.4
                
                # Boost score for exact category name matches
                if category_name.lower() in text_fields['name'].lower():
                    final_score *= 1.5
                
                # Log detailed scoring for debugging
                self.logger.debug(f"\nScoring details for {category_name}:")
                self.logger.debug(f"Base score: {base_score:.3f}")
                self.logger.debug(f"Density score: {avg_density:.3f}")
                self.logger.debug(f"Diversity score: {field_diversity:.3f}")
                self.logger.debug(f"Final score: {final_score:.3f}")
                self.logger.debug(f"Matched keywords: {matched_keywords}")
                self.logger.debug(f"Field matches: {field_matches}")
                
                # Update best match if this score is higher
                if final_score > best_match[1]:
                    best_match = (
                        category_name,
                        final_score,
                        matched_keywords,
                        {
                            'field_matches': field_matches,
                            'term_weights': term_weights,
                            'density_score': avg_density,
                            'diversity_score': field_diversity,
                            'base_score': base_score
                        }
                    )
        
        # Return results with category name if score is high enough
        return {
            'keyword_score': best_match[1],
            'matched_keywords': best_match[2],
            'category_name': best_match[0] if best_match[1] > 0.25 else None,
            'match_details': best_match[3] if best_match[1] > 0.25 else {},
            '_best_match_category': best_match[0]  # Store best match category regardless of threshold
        }

    def _perform_semantic_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic analysis using sentence transformers"""
        # Initialize text processor if not already done
        if not hasattr(self, 'text_processor'):
            self.text_processor = TextProcessor()
            
        # Format use case text as a dictionary
        use_case_text = {
            'description': use_case.get('description', ''),
            'purpose_benefits': use_case.get('purpose_benefits', ''),
            'outputs': use_case.get('outputs', ''),
            'systems': use_case.get('name', '')  # Using name as systems field
        }
        
        results = {}
        
        # Check each category
        for category_name, category in self.categories.items():
            # Format category text as a dictionary
            category_text = {
                'description': f"{category.name}: {' '.join(category.keywords)}",
                'purpose_benefits': ' '.join(category.business_language),
                'outputs': ' '.join(category.capabilities),
                'systems': category.name
            }
            
            score = self.text_processor.calculate_weighted_semantic_score(
                use_case_text=use_case_text,
                category_text=category_text
            )
            
            results[category_name] = {
                'score': score,
                'semantic_score': score,
                'confidence': score if score > 0.30 else 0.0
            }
        
        return results

    async def _perform_llm_analysis(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM-based analysis comparing the use case against all AI categories.
        Uses a structured approach considering all relevant use case fields and category details.
        """
        try:
            # Format use case text with all relevant fields
            use_case_text = self._format_use_case_text(use_case)
            
            # Get all active categories formatted for comparison
            categories_text = self._format_categories_for_prompt()
            
            # Prepare the analysis prompt
            messages = [
                {"role": "system", "content": """You are an expert AI technology analyst evaluating federal use cases.
Your task is to determine the best matching AI technology category for this use case based on detailed analysis.

Provide your response in JSON format with the following structure:
{
    "best_match": {
        "category_name": string,       // Name of best matching category
        "confidence": float,           // Confidence score (0.0-1.0)
        "match_type": string,          // "PRIMARY", "SUPPORTING", "RELATED", or "NONE"
        "reasoning": string            // Detailed explanation of the match
    },
    "field_analysis": {
        "technical_alignment": float,   // Score for technical alignment (0.0-1.0)
        "business_alignment": float,    // Score for business alignment (0.0-1.0)
        "implementation_fit": float,    // Score for implementation approach fit (0.0-1.0)
        "capability_coverage": float,   // Score for capability coverage (0.0-1.0)
        "maturity_alignment": float     // Score for maturity level alignment (0.0-1.0)
    },
    "matched_terms": {
        "technical_terms": string[],    // Matched technical terms
        "business_terms": string[],     // Matched business terms
        "capabilities": string[]        // Matched capabilities
    },
    "alternative_matches": [{           // Up to 2 alternative matches
        "category_name": string,
        "confidence": float,
        "match_type": string,
        "reasoning": string
    }]
}"""},
                {"role": "user", "content": f"""
EVALUATION CRITERIA:
1. Technical Alignment:
   - Does the use case implementation align with category core capabilities?
   - Are the required technical components present?
   - Do the outputs match expected category outcomes?

2. Business Alignment:
   - Does the use case purpose align with category business objectives?
   - Are the benefits consistent with category capabilities?
   - Is the maturity level appropriate?

3. Implementation Context:
   - Is the development approach consistent with the category?
   - Does the system architecture fit the category pattern?
   - Are there any technical conflicts or misalignments?

USE CASE DETAILS:
{use_case_text}

AVAILABLE AI TECHNOLOGY CATEGORIES:
{categories_text}

Analyze this use case against all available categories and provide your assessment in the specified JSON format.
Focus on finding the best matching category based on both technical and business alignment.
If no strong match is found, set match_type to "NONE" and explain why in the reasoning."""}
            ]

            # Get LLM analysis
            response = await self.llm_analyzer._call_openai(messages)
            
            # Extract the best match details
            best_match = response.get("best_match", {})
            field_scores = response.get("field_analysis", {})
            matched_terms = response.get("matched_terms", {})
            
            # Calculate overall LLM score as weighted average of field scores
            weights = {
                "technical_alignment": 0.3,
                "business_alignment": 0.3,
                "implementation_fit": 0.2,
                "capability_coverage": 0.1,
                "maturity_alignment": 0.1
            }
            
            llm_score = sum(
                field_scores.get(field, 0.0) * weight 
                for field, weight in weights.items()
            )
            
            # Apply term match boost
            term_match_boost = 0.0
            if matched_terms:
                technical_terms = len(matched_terms.get("technical_terms", []))
                business_terms = len(matched_terms.get("business_terms", []))
                capabilities = len(matched_terms.get("capabilities", []))
                
                # Add 5% boost for each matched term up to 20%
                term_match_boost = min(0.2, (technical_terms + business_terms + capabilities) * 0.05)
                
            llm_score = min(1.0, llm_score * (1 + term_match_boost))
            
            return {
                'llm_score': llm_score,
                'category_name': best_match.get("category_name"),
                'match_type': best_match.get("match_type", "NONE"),
                'llm_explanation': best_match.get("reasoning", ""),
                'field_match_scores': field_scores,
                'matched_terms': matched_terms,
                'alternative_matches': response.get("alternative_matches", [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                'llm_score': 0.0,
                'llm_explanation': f"Error during LLM analysis: {str(e)}"
            }

    def _format_use_case_text(self, use_case: Dict[str, Any]) -> str:
        """Format use case data for LLM analysis."""
        sections = []
        
        # Add name
        if use_case.get('name'):
            sections.append(f"Name: {use_case['name']}")
            
        # Add description
        if use_case.get('description'):
            sections.append(f"Description: {use_case['description']}")
            
        # Add purpose and benefits
        if use_case.get('purpose_benefits'):
            sections.append(f"Purpose and Benefits: {use_case['purpose_benefits']}")
            
        # Add outputs
        if use_case.get('outputs'):
            sections.append(f"Outputs: {use_case['outputs']}")
            
        # Add topic area
        if use_case.get('topic_area'):
            sections.append(f"Topic Area: {use_case['topic_area']}")
            
        # Add stage
        if use_case.get('stage'):
            sections.append(f"Stage: {use_case['stage']}")
            
        # Add impact type
        if use_case.get('impact_type'):
            sections.append(f"Impact Type: {use_case['impact_type']}")
            
        # Add development method
        if use_case.get('dev_method'):
            sections.append(f"Development Method: {use_case['dev_method']}")
            
        # Add system name
        if use_case.get('system_name'):
            sections.append(f"System Name: {use_case['system_name']}")
            
        return "\n\n".join(sections)

    def _combine_analysis_results(self, results: Dict[str, Any]) -> None:
        """Combine results from different analysis methods and determine match types"""
        # Get individual scores
        keyword_score = results['keyword_score']
        semantic_score = results['semantic_score']
        llm_score = results['llm_score']
        
        # Get detailed field scores if available
        field_scores = results.get('field_match_scores', {})
        
        # Calculate enhanced confidence score using all available metrics
        confidence_components = {
            'keyword': keyword_score * 0.20,
            'semantic': semantic_score * 0.20,
            'llm': llm_score * 0.15,
            'technical': field_scores.get('technical_alignment', 0.0) * 0.10,
            'business': field_scores.get('business_alignment', 0.0) * 0.10,
            'implementation': field_scores.get('implementation_fit', 0.0) * 0.05,
            'capability': field_scores.get('capability_coverage', 0.0) * 0.05,
            'semantic_relevance': field_scores.get('semantic_relevance', 0.0) * 0.05,
            'keyword_relevance': field_scores.get('keyword_relevance', 0.0) * 0.05,
            'context_alignment': field_scores.get('context_alignment', 0.0) * 0.05
        }
        
        # Calculate weighted confidence score
        combined_score = sum(confidence_components.values())
        
        # Apply method agreement boost
        method_agreement = sum(1 for score in [
            keyword_score,
            semantic_score,
            llm_score,
            field_scores.get('technical_alignment', 0.0),
            field_scores.get('business_alignment', 0.0)
        ] if score > 0.4)
        
        if method_agreement > 1:
            combined_score *= (1 + (method_agreement - 1) * 0.1)  # 10% boost per agreeing method
            
        # Determine match type based on combined score and individual components
        if (combined_score > 0.45 and 
            min(keyword_score, semantic_score) > 0.3 and 
            field_scores.get('technical_alignment', 0.0) > 0.4 and
            field_scores.get('context_alignment', 0.0) > 0.35):
            # Strong match with good technical alignment and context
            results['match_type'] = 'PRIMARY'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 1
            
        elif (combined_score > 0.35 or 
              max(keyword_score, semantic_score) > 0.4 or 
              field_scores.get('business_alignment', 0.0) > 0.5 or
              field_scores.get('semantic_relevance', 0.0) > 0.45):
            # Good business alignment or strong individual scores
            results['match_type'] = 'SUPPORTING'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 2
            
        elif (combined_score > 0.25 or 
              any(score > 0.3 for score in [
                  keyword_score,
                  semantic_score,
                  llm_score,
                  field_scores.get('semantic_relevance', 0.0),
                  field_scores.get('context_alignment', 0.0)
              ])):
            # Some relevance detected
            results['match_type'] = 'RELATED'
            results['match_method'] = 'ENSEMBLE'
            results['match_rank'] = 3
            
        else:
            results['match_type'] = 'NONE'
            results['match_method'] = 'INSUFFICIENT_CONFIDENCE'
            results['match_rank'] = 0
        
        # Update confidence score
        results['confidence'] = round(combined_score, 3)
        
        # Set category name based on match type and confidence
        if results['match_type'] != 'NONE':
            results['category_name'] = results.get('_best_match_category')
            results['relationship_type'] = 'USES_TECHNOLOGY'
        else:
            results['category_name'] = None
            results['relationship_type'] = 'NO_MATCH'
        
        # Enhance match details with semantic concepts if available
        term_match_details = results.get('term_match_details', {})
        if 'semantic_concepts' in term_match_details:
            semantic_concepts = term_match_details['semantic_concepts']
            # Add high-relevance concepts to matched terms
            for concept_type in ['technical_concepts', 'business_concepts']:
                for concept in semantic_concepts.get(concept_type, []):
                    if concept['relevance'] > 0.7:  # Only include high-relevance concepts
                        term_match_details.setdefault('context_matches', []).append(
                            f"{concept['concept']} ({concept['relevance']:.2f})"
                        )
        
        # Update v2.2 schema properties
        results.update({
            'field_match_scores': field_scores,
            'term_match_details': term_match_details,
            'analysis_version': 'v2.2',
            'review_status': 'PENDING',
            'false_positive': False,
            'manual_override': False
        })
        
        # Log combination details for analysis
        self.logger.debug(
            f"Score combination: keyword={keyword_score:.3f}, "
            f"semantic={semantic_score:.3f}, "
            f"llm={llm_score:.3f}, "
            f"combined={combined_score:.3f}, "
            f"match_type={results['match_type']}, "
            f"method={results['match_method']}"
        )

    def verify_match(self, use_case_text: str, category_name: str, match_type: str) -> float:
        """Verify a potential match using LLM."""
        try:
            response = self.llm_analyzer.verify_match(use_case_text, category_name, match_type)
            if isinstance(response, str):
                response_json = json.loads(response)
                confidence = float(response_json.get('confidence', 0.0))
                return confidence
            return 0.0
        except Exception as e:
            self.logger.error(f"LLM verification failed: {str(e)}")
            return 0.0

    def analyze_unmatched_case(self, use_case_text: str) -> Dict[str, Any]:
        """Analyze an unmatched use case using LLM."""
        try:
            response = self.llm_analyzer.analyze_unmatched(use_case_text)
            if isinstance(response, str):
                response_json = json.loads(response)
                return {
                    'reason': response_json.get('reason', 'Unknown'),
                    'reason_category': response_json.get('reason_category', 'Unknown'),
                    'suggestions': response_json.get('suggestions', [])
                }
            return {
                'reason': 'Error during analysis',
                'reason_category': 'Error',
                'suggestions': []
            }
        except Exception as e:
            self.logger.error(f"Error analyzing unmatched case: {str(e)}")
            return {
                'reason': 'Error during analysis',
                'reason_category': 'Error',
                'suggestions': []
            }

    def _format_categories_for_prompt(self) -> str:
        """Format AI categories for LLM prompt"""
        formatted_categories = []
        for category in self.categories.values():
            formatted = f"""
Category: {category.name}
Definition: {category.definition}
Keywords: {', '.join(category.keywords) if category.keywords else 'N/A'}
Capabilities: {', '.join(category.capabilities) if category.capabilities else 'N/A'}
Business Terms: {', '.join(category.business_language) if category.business_language else 'N/A'}
Maturity Level: {category.maturity_level}
"""
            formatted_categories.append(formatted)
        return "\n".join(formatted_categories)

    async def analyze_no_match(self, use_case: Dict[str, Any], current_category: Optional[str] = None) -> Dict[str, Any]:
        """Analyze cases with no strong matches using LLM."""
        try:
            # Format use case text
            use_case_text = self._format_use_case_text(use_case)
            
            # Get LLM analysis
            llm_results = await self.llm_analyzer.analyze_no_match(use_case_text, current_category)
            
            # Format results
            return {
                'category': llm_results.get('suggested_category'),
                'confidence': llm_results.get('confidence', 0.0),
                'explanation': llm_results.get('reasoning', ''),
                'suggestions': llm_results.get('suggestions', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                'category': None,
                'confidence': 0.0,
                'explanation': f"Error during analysis: {str(e)}",
                'suggestions': []
            }

    async def get_random_use_cases(self, num_cases: int) -> List[Dict]:
        """Get random use cases from Neo4j that haven't been classified yet."""
        try:
            async with self.driver.session() as session:
                # Try to get unclassified use cases first
                query = """
                MATCH (u:UseCase)
                WHERE NOT EXISTS((u)-[:CLASSIFIED_AS]->(:AIClassification))
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
                result = await session.run(query, limit=num_cases)
                use_cases = [record["use_case"] async for record in result]
                
                # If we don't have enough unclassified cases, get any random cases
                if len(use_cases) < num_cases:
                    remaining = num_cases - len(use_cases)
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
                    result = await session.run(query, limit=remaining)
                    additional_cases = [record["use_case"] async for record in result]
                    use_cases.extend(additional_cases)
                
                return use_cases
                
        except Exception as e:
            logger.error(f"Error fetching random use cases: {str(e)}")
            return [] 