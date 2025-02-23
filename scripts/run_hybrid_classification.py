"""
Dell-AITC Hybrid Classification Runner (v2.2)
Combines keyword-based and LLM-based classification with optimized batching.

Usage:
    python scripts/run_hybrid_classification.py [--batch-size N] [--max-cases N] [--dry-run]
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple
import argparse
from tqdm import tqdm
import openai

from backend.app.services.ai.classifier import Classifier
from backend.app.services.ai.keyword_classifier import KeywordClassifier
from backend.app.services.database.verifier import DatabaseVerifier
from backend.app.config import get_settings
from backend.app.services.ai.llm_service import LLMService

# Configure logging
log_dir = Path("logs/classification")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'hybrid_classification_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridClassifier:
    """Combines keyword and LLM classification methods"""
    
    def __init__(self, batch_size: int = 10, dry_run: bool = True):
        """Initialize the hybrid classifier.
        
        Args:
            batch_size: Number of use cases to process in each batch
            dry_run: If True, don't save results to database
        """
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.base_classifier = None
        self.keyword_classifier = None
        self.verifier = None
        self.settings = get_settings()
        
        # Configure confidence thresholds
        self.high_confidence = 0.8
        self.medium_confidence = 0.6
        
        # Configure output paths
        self.output_dir = Path("data/output/hybrid_classification")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure OpenAI
        openai.api_key = self.settings.openai_api_key
        self.llm_model = "gpt-4o-mini"
        
    async def __aenter__(self):
        """Initialize classifiers and verifier."""
        # Initialize base classifier
        self.base_classifier = Classifier()
        await self.base_classifier.initialize()

        # Initialize keyword classifier
        self.keyword_classifier = KeywordClassifier()
        await self.keyword_classifier.initialize()

        # Initialize verifier with API key
        self.verifier = LLMService(api_key=self.settings.openai_api_key)
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        if self.base_classifier:
            await self.base_classifier.close()
        
        if self.keyword_classifier:
            await self.keyword_classifier.cleanup()
            
        if self.verifier:
            await self.verifier.cleanup()
        
    async def verify_with_llm(self, use_case: Dict[str, Any], category: str, confidence: float) -> Tuple[bool, float, str]:
        """Verify a medium confidence match using LLM.
        
        Args:
            use_case: Use case dictionary
            category: Category name to verify
            confidence: Initial confidence score
            
        Returns:
            Tuple of (is_verified, new_confidence, explanation)
        """
        # Combine use case text
        use_case_text = f"""
        Use Case: {use_case.get('name', '')}
        Description: {use_case.get('description', '')}
        Purpose/Benefits: {use_case.get('purpose_benefits', '')}
        Outputs: {use_case.get('outputs', '')}
        """
        
        # Get category details
        async with self.base_classifier.driver.session() as session:
            result = await session.run("""
                MATCH (c:AICategory {name: $category})
                RETURN c.category_definition as definition,
                       c.maturity_level as maturity,
                       c.capabilities as capabilities
                """, category=category)
            cat_data = await result.single()
            
        category_text = f"""
        Category: {category}
        Definition: {cat_data['definition']}
        Maturity: {cat_data['maturity']}
        Capabilities: {', '.join(cat_data['capabilities'] or [])}
        """
        
        # Prepare prompt for verification
        prompt = f"""You are verifying if a federal AI use case matches an AI technology category.
        Initial keyword-based confidence: {confidence:.2f}
        
        {category_text}
        
        {use_case_text}
        
        Analyze the alignment between the use case and category. Consider:
        1. Technical alignment with category definition
        2. Required capabilities match
        3. Implementation feasibility
        4. Maturity level appropriateness
        
        Provide your analysis in JSON format:
        {{
            "is_verified": boolean,
            "confidence_adjustment": float,  # Between -0.2 and +0.2
            "explanation": string,
            "key_points": [string]
        }}
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            new_confidence = min(1.0, max(0.0, confidence + result['confidence_adjustment']))
            
            return (result['is_verified'], new_confidence, result['explanation'])
            
        except Exception as e:
            self.logger.error(f"Error in LLM verification: {str(e)}")
            return (False, confidence, f"LLM verification failed: {str(e)}")
            
    async def analyze_with_llm(self, use_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform full LLM analysis for low/no confidence matches.
        
        Args:
            use_case: Use case dictionary
            
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        # Get all categories
        async with self.base_classifier.driver.session() as session:
            result = await session.run("""
                MATCH (c:AICategory)
                WHERE c.status = 'active'
                RETURN c.name as name,
                       c.category_definition as definition,
                       c.maturity_level as maturity
                ORDER BY c.name
                """)
            categories = [dict(record) async for record in result]
            
        # Prepare categories text
        categories_text = "\n\n".join([
            f"Category: {cat['name']}\n"
            f"Definition: {cat['definition']}\n"
            f"Maturity: {cat['maturity']}"
            for cat in categories
        ])
        
        # Prepare use case text
        use_case_text = f"""
        Use Case: {use_case.get('name', '')}
        Description: {use_case.get('description', '')}
        Purpose/Benefits: {use_case.get('purpose_benefits', '')}
        Outputs: {use_case.get('outputs', '')}
        """
        
        prompt = f"""You are analyzing a federal AI use case to determine the most appropriate AI technology categories.
        Review the use case against all available categories and identify the best matches.
        
        Use Case:
        {use_case_text}
        
        Available Categories:
        {categories_text}
        
        Analyze the use case and provide your assessment in JSON format:
        {{
            "primary_category": {{
                "name": string,
                "confidence": float,
                "reasoning": string
            }},
            "alternative_categories": [
                {{
                    "name": string,
                    "confidence": float,
                    "reasoning": string
                }}
            ],
            "analysis": {{
                "key_technologies": [string],
                "implementation_considerations": string,
                "confidence_factors": string
            }}
        }}
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return None
            
    async def process_batch(self, use_cases: List[Dict]) -> None:
        """Process a batch of use cases."""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Processing batch of {len(use_cases)} use cases")
        self.logger.info("="*80 + "\n")
        
        for i, use_case in enumerate(use_cases, 1):
            self.logger.info(f"\nProcessing use case {i}/{len(use_cases)}")
            self.logger.info(f"ID: {use_case.get('id')}")
            self.logger.info(f"Name: {use_case.get('name')}")
            
            # Combine text fields for classification
            text = " ".join(filter(None, [
                use_case.get('name', ''),
                use_case.get('description', ''),
                use_case.get('purpose_benefits', ''),
                use_case.get('outputs', '')
            ]))
            
            if not text.strip():
                self.logger.warning(f"Empty text for use case {use_case.get('id')}")
                continue
                
            self.logger.info("\n1. Attempting keyword classification...")
            # First try keyword classification
            keyword_results = self.keyword_classifier.classify_text(text)
            
            if not keyword_results:
                self.logger.info("No keyword matches found - proceeding to full LLM analysis")
                # Perform full LLM analysis
                self.logger.info("\n2. Performing full LLM analysis...")
                llm_analysis = await self.analyze_with_llm(use_case)
                if llm_analysis:
                    self.logger.info(f"LLM Analysis Results:")
                    self.logger.info(f"Primary Category: {llm_analysis['primary_category']['name']}")
                    self.logger.info(f"Confidence: {llm_analysis['primary_category']['confidence']:.2f}")
                    self.logger.info(f"Reasoning: {llm_analysis['primary_category']['reasoning']}")
                    
                    if llm_analysis['primary_category']['confidence'] >= self.medium_confidence:
                        category = llm_analysis['primary_category']['name']
                        confidence = llm_analysis['primary_category']['confidence']
                        if not self.dry_run:
                            await self.verifier.save_classification(
                                use_case["id"],
                                category,
                                confidence,
                                "llm_full"
                            )
                        self.logger.info(f"Classification saved: {category} ({confidence:.2f})")
                    else:
                        self.logger.info("Confidence too low to save classification")
                else:
                    self.logger.warning("LLM analysis failed")
                continue
                
            # Get best match
            best_match = keyword_results[0].get_best_match()
            if not best_match:
                continue
                
            confidence = best_match.get('confidence', 0.0)
            category = best_match.get('category_name')
            
            self.logger.info(f"\nKeyword Classification Results:")
            self.logger.info(f"Best Match: {category}")
            self.logger.info(f"Confidence: {confidence:.2f}")
            
            if confidence >= self.high_confidence:
                self.logger.info("High confidence match - saving directly")
                if not self.dry_run:
                    await self.verifier.save_classification(
                        use_case["id"], 
                        category,
                        confidence,
                        "keyword"
                    )
            elif confidence >= self.medium_confidence:
                self.logger.info("\n2. Verifying medium confidence match with LLM...")
                # Verify with LLM
                is_verified, new_confidence, explanation = await self.verify_with_llm(
                    use_case, category, confidence
                )
                self.logger.info(f"LLM Verification Results:")
                self.logger.info(f"Verified: {is_verified}")
                self.logger.info(f"New Confidence: {new_confidence:.2f}")
                self.logger.info(f"Explanation: {explanation}")
                
                if is_verified:
                    if not self.dry_run:
                        await self.verifier.save_classification(
                            use_case["id"],
                            category,
                            new_confidence,
                            "keyword_llm_verified"
                        )
                    self.logger.info("Verification successful - saving classification")
                else:
                    self.logger.info("\n3. Match not verified - attempting full LLM analysis...")
                    # If not verified, try full LLM analysis
                    llm_analysis = await self.analyze_with_llm(use_case)
                    if llm_analysis and llm_analysis['primary_category']['confidence'] >= self.medium_confidence:
                        category = llm_analysis['primary_category']['name']
                        confidence = llm_analysis['primary_category']['confidence']
                        if not self.dry_run:
                            await self.verifier.save_classification(
                                use_case["id"],
                                category,
                                confidence,
                                "llm_full"
                            )
                        self.logger.info(f"New classification from LLM: {category} ({confidence:.2f})")
                    else:
                        self.logger.info("No suitable classification found from LLM analysis")
            else:
                self.logger.info("\n2. Low confidence match - proceeding to full LLM analysis...")
                # Perform full LLM analysis
                llm_analysis = await self.analyze_with_llm(use_case)
                if llm_analysis and llm_analysis['primary_category']['confidence'] >= self.medium_confidence:
                    category = llm_analysis['primary_category']['name']
                    confidence = llm_analysis['primary_category']['confidence']
                    if not self.dry_run:
                        await self.verifier.save_classification(
                            use_case["id"],
                            category,
                            confidence,
                            "llm_full"
                        )
                    self.logger.info(f"Classification from LLM: {category} ({confidence:.2f})")
                else:
                    self.logger.info("No suitable classification found from LLM analysis")
            
            self.logger.info("\n" + "-"*80)

    async def run(self, max_cases: Optional[int] = None) -> None:
        """Run the hybrid classification process.
        
        Args:
            max_cases: Optional maximum number of cases to process
        """
        try:
            # Get unclassified use cases
            use_cases = await self.base_classifier.get_unclassified_use_cases(limit=max_cases)
            if not use_cases:
                self.logger.info("No unclassified use cases found.")
                return
                
            # Process in batches
            for i in range(0, len(use_cases), self.batch_size):
                batch = use_cases[i:i + self.batch_size]
                await self.process_batch(batch)
                
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.output_dir / f"metrics_{timestamp}.json"
            # TODO: Add metrics collection and saving
            
        except Exception as e:
            self.logger.error(f"Error during hybrid classification: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run hybrid classification on use cases")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Don't save results to database")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases to process")
    return parser.parse_args()

async def main():
    """Main entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run hybrid classification on use cases")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Don't save results to database")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases to process")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting hybrid classification with batch size {args.batch_size}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be saved to database")
    
    logger.info("Initializing hybrid classifier...")
    async with HybridClassifier(batch_size=args.batch_size, dry_run=args.dry_run) as classifier:
        await classifier.run(max_cases=args.max_cases)

if __name__ == "__main__":
    asyncio.run(main()) 