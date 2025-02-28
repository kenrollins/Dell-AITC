"""
Enhanced vendor analysis script that processes partners from a CSV file and generates analysis results.

Usage:
    python enhanced_vendor_analysis.py [--limit N] [--csv-file PATH] [--confidence-threshold FLOAT]
"""

import argparse
import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import aiohttp
import httpx
from bs4 import BeautifulSoup
from backend.app.services.database.neo4j_service import Neo4jService

# Configure logging with more detail
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vendor_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Define confidence threshold and Ollama settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_OLLAMA_URL = "http://kaiju.home.arpa:11434"
DEFAULT_OLLAMA_MODEL = "command-r-plus:104b"

class CategoryAnalysis:
    """Structured category analysis results."""
    def __init__(self, category_name: str):
        self.category_name = category_name
        self.confidence_score = 0.0
        self.evidence = []
        self.capabilities = []
        self.technical_keywords = []
        self.business_terms = []
        self.summary = ""
        self.implementation_details = ""
        self.integration_points = []
        self.github_evidence = []
        
    def to_dict(self) -> Dict:
        """Convert analysis to dictionary format."""
        return {
            "category_name": self.category_name,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "capabilities": self.capabilities,
            "technical_keywords": self.technical_keywords,
            "business_terms": self.business_terms,
            "summary": self.summary,
            "implementation_details": self.implementation_details,
            "integration_points": self.integration_points,
            "github_evidence": self.github_evidence
        }

class OllamaClient:
    """Client for interacting with Ollama API with improved error handling."""
    
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = httpx.AsyncClient(timeout=timeout)
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.aclose()
        
    async def check_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = await self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            if not any(m['name'] == self.model for m in models):
                self.logger.error(f"Model {self.model} not found on server. Available models: {[m['name'] for m in models]}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False
            
    async def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate text with retries and detailed error handling."""
        if not await self.check_connection():
            return None
            
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Making Ollama API request (attempt {attempt + 1}/{max_retries})")
                response = await self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response')
                
            except Exception as e:
                # Check if it's a timeout exception (different ways to detect it)
                if "timeout" in str(e).lower() or isinstance(e, httpx.ReadTimeout) or isinstance(e, httpx.ConnectTimeout):
                    self.logger.error(f"Request timed out (attempt {attempt + 1})")
                elif isinstance(e, httpx.HTTPError):
                    self.logger.error(f"HTTP error occurred: {str(e)} (attempt {attempt + 1})")
                else:
                    self.logger.error(f"Unexpected error: {str(e)} (attempt {attempt + 1})")
                    
                if attempt == max_retries - 1:
                    return None
                    
            await asyncio.sleep(1)  # Wait before retry
        
        return None

class WebsiteAnalyzer:
    """Analyzes website content."""
    
    def __init__(self):
        self.session = None
        
    async def initialize(self):
        """Initialize the HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def analyze_website(self, url: str) -> Dict:
        """Analyze website content."""
        try:
            await self.initialize()
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {"error": f"Failed to fetch website: {response.status}"}
                    
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract relevant information
                title = soup.title.string if soup.title else ""
                description = ""
                desc_tag = soup.find("meta", attrs={"name": "description"})
                if desc_tag:
                    description = desc_tag.get("content", "")
                    
                # Extract text content - improved to get more comprehensive content
                # Get text from paragraphs, headings, list items, and divs with text
                text_elements = []
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
                    # Skip empty elements or those with just whitespace
                    if tag.get_text(strip=True):
                        # Skip nested elements to avoid duplication
                        if not any(parent in text_elements for parent in tag.parents):
                            text_elements.append(tag)
                
                text_content = " ".join([elem.get_text(strip=True) for elem in text_elements])
                
                # Extract structured data if available
                structured_data = {}
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        data = json.loads(script.string)
                        if data:
                            structured_data = data
                            break
                    except:
                        pass
                
                return {
                    "title": title,
                    "description": description,
                    "content": text_content[:7500],  # Increased limit for more context
                    "structured_data": structured_data
                }
        except Exception as e:
            logger.error(f"Error analyzing website: {e}")
            return {"error": str(e)}

class MultiSourceSearcher:
    """Handles searching across multiple sources (GitHub, web) for vendor information."""
    
    def __init__(self, vendor_name: str, ollama_client: OllamaClient):
        self.vendor_name = vendor_name
        self.ollama_client = ollama_client
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def search_github(self, query: str) -> List[Dict]:
        """Search GitHub repositories with enhanced queries and information extraction."""
        results = []
        
        # Ensure query is not None and is a string
        if query is None:
            logger.warning("GitHub search received None query")
            query = ""
        
        # Ensure vendor_name is not None
        vendor_name = self.vendor_name if self.vendor_name is not None else ""
        
        # Create multiple search queries for better coverage
        search_queries = []
        
        # Basic search if query is not empty
        if query.strip():
            search_queries.append(f"{query} in:name,description,readme")
            
            # Only add this if we have both vendor name and query
            if vendor_name.strip() and len(query.split()) > 0:
                search_queries.append(f"{vendor_name} {query.split()[0]} in:name,description,readme")
        
        # Always add this general search if we have a vendor name
        if vendor_name.strip():
            search_queries.append(f"{vendor_name} AI technology in:readme")
        
        # If we have no valid queries, return empty results
        if not search_queries:
            logger.warning("No valid GitHub search queries could be constructed")
            return []
        
        try:
            for search_query in search_queries:
                logger.debug(f"Searching GitHub with query: {search_query}")
                async with self.session.get(
                    "https://api.github.com/search/repositories",
                    params={"q": search_query, "sort": "stars", "order": "desc"},
                    headers={"Accept": "application/vnd.github.v3+json"}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"GitHub API returned status {response.status} for query: {search_query}")
                        continue
                        
                    data = await response.json()
                    
                    # Process each repository
                    for repo in data.get("items", [])[:3]:  # Top 3 results per query
                        # Skip if we already have this repo
                        if any(r["title"] == repo["name"] for r in results):
                            continue
                            
                        # Get additional repo details - languages used
                        languages = {}
                        try:
                            async with self.session.get(
                                repo["languages_url"],
                                headers={"Accept": "application/vnd.github.v3+json"}
                            ) as lang_response:
                                if lang_response.status == 200:
                                    languages = await lang_response.json()
                        except Exception as e:
                            logger.warning(f"Error fetching languages for {repo['name']}: {e}")
                        
                        # Get README content for more context
                        readme_content = ""
                        try:
                            async with self.session.get(
                                f"https://api.github.com/repos/{repo['full_name']}/readme",
                                headers={"Accept": "application/vnd.github.v3.raw"}
                            ) as readme_response:
                                if readme_response.status == 200:
                                    readme_content = await readme_response.text()
                                    # Limit readme length
                                    readme_content = readme_content[:1000] + "..." if len(readme_content) > 1000 else readme_content
                        except Exception as e:
                            logger.warning(f"Error fetching README for {repo['name']}: {e}")
                        
                        results.append({
                            "title": repo["name"],
                            "full_name": repo.get("full_name", ""),
                            "description": repo.get("description", ""),
                            "stars": repo.get("stargazers_count", 0),
                            "forks": repo.get("forks_count", 0),
                            "link": repo["html_url"],
                            "languages": languages,
                            "readme_excerpt": readme_content,
                            "last_updated": repo.get("updated_at", ""),
                            "created_at": repo.get("created_at", ""),
                            "relevance_score": self._calculate_relevance(repo, query)
                        })
            
            # Sort by relevance score and limit to top 5 overall
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:5]
            
        except Exception as e:
            logger.error(f"GitHub search error: {str(e)}")
            return []
    
    def _calculate_relevance(self, repo: Dict, query: str) -> float:
        """Calculate relevance score for a repository based on various factors."""
        score = 0.0
        
        # Base score from stars (max 5 points)
        stars = repo.get("stargazers_count", 0)
        if stars > 1000:
            score += 5.0
        elif stars > 500:
            score += 4.0
        elif stars > 100:
            score += 3.0
        elif stars > 50:
            score += 2.0
        elif stars > 10:
            score += 1.0
        
        # Recent activity bonus (max 2 points)
        try:
            from datetime import datetime
            last_updated = datetime.strptime(repo.get("updated_at", ""), "%Y-%m-%dT%H:%M:%SZ")
            days_since_update = (datetime.now() - last_updated).days
            if days_since_update < 30:
                score += 2.0
            elif days_since_update < 90:
                score += 1.0
            elif days_since_update < 180:
                score += 0.5
        except:
            pass
        
        # Keyword relevance in name/description (max 3 points)
        name_desc = (repo.get("name", "") + " " + repo.get("description", "")).lower()
        query_terms = query.lower().split()
        matches = sum(1 for term in query_terms if term in name_desc)
        score += min(3.0, matches * 0.5)
        
        # Vendor name in repo (bonus 2 points)
        if self.vendor_name.lower() in name_desc:
            score += 2.0
            
        return score

    async def search_for_category(self, category: Dict, website_analysis: Dict) -> List[Dict]:
        """Search for evidence of category capabilities with improved extraction and error handling."""
        # Combine website content with GitHub results
        evidence = []
        
        try:
            # Validate inputs
            if category is None:
                logger.warning("Category is None in search_for_category")
                return evidence
                
            if website_analysis is None:
                logger.warning("Website analysis is None in search_for_category")
                return evidence
            
            # Get category name with fallback
            category_name = category.get('name', 'Unknown Category')
            
            # Extract relevant sections from website
            content = website_analysis.get("content", "")
            if not content:
                logger.warning(f"No content available for category {category_name}")
                return evidence
                
            # Get category attributes with safe defaults
            category_definition = category.get('category_definition', 'No definition available')
            technical_keywords = category.get('technical_keywords', [])
            business_terms = category.get('business_terms', [])
            capabilities = category.get('capabilities', [])
            
            # Ensure lists are actually lists
            if not isinstance(technical_keywords, list):
                technical_keywords = []
            if not isinstance(business_terms, list):
                business_terms = []
            if not isinstance(capabilities, list):
                capabilities = []
            
            # Use Ollama to extract relevant sections with improved prompt
            prompt = f"""Given the following website content, identify sections that demonstrate capabilities in {category_name} technology category.

Category Definition: {category_definition}

Technical Keywords to look for: {', '.join(technical_keywords)}
Business Terms to look for: {', '.join(business_terms)}
Core Capabilities to identify: {', '.join(capabilities)}

Website Content to Analyze:
{content[:3500]}

For each relevant section you find, extract the exact text and assign a relevance score (0.0-1.0).
Higher scores should be given to sections that:
1. Explicitly mention capabilities related to {category_name}
2. Use technical terminology from the keywords list
3. Describe specific implementations or use cases
4. Provide details about integration or deployment

Return a JSON array of relevant sections, each with:
[
  {{
    "text": "extracted text (keep this concise and focused on the evidence)",
    "relevance": float (0.0-1.0),
    "source": "Content" or other appropriate source identifier
  }}
]

Only include high-confidence findings with clear evidence. Limit to the 3-5 most relevant sections.
"""
            
            response = await self.ollama_client.generate(prompt)
            if response:
                try:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        sections = json.loads(json_str)
                    else:
                        # Fallback to direct parsing
                        sections = json.loads(response)
                        
                    if isinstance(sections, list):
                        evidence.extend([
                            {**section, "source": section.get("source", "Website Content")}
                            for section in sections
                        ])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Ollama response for category search: {e}")
                    logger.debug(f"Raw response: {response[:500]}...")
                    
            # Add structured data evidence if available
            structured_data = website_analysis.get("structured_data", {})
            if structured_data and isinstance(structured_data, dict):
                # Extract relevant information from structured data
                for key, value in structured_data.items():
                    if isinstance(value, str) and any(kw.lower() in value.lower() for kw in technical_keywords if isinstance(kw, str)):
                        evidence.append({
                            "text": f"{key}: {value}",
                            "relevance": 0.8,
                            "source": "Structured Data"
                        })
        except Exception as e:
            logger.error(f"Error in search_for_category for {category.get('name', 'unknown')}: {str(e)}")
            
        return evidence

def load_partners(csv_path: str = "data/input/Dell Federal AI Partner Tracking.csv", limit: Optional[int] = None) -> List[Dict]:
    """Load partner data from CSV file."""
    logger.info(f"Loading partners from {csv_path} with limit {limit}")
    partners = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            # Skip schema definition line
            next(f)
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                logger.debug(f"Processing row {i}: {row}")
                if limit and i >= limit:
                    break
                    
                name = row.get("Partner Name", "").strip()
                website = row.get("Company Website", "").strip()  # Updated column name
                
                logger.debug(f"Found partner: {name} with website: {website}")
                
                # Clean up SharePoint URLs if present
                if "sharepoint.com" in website.lower():
                    logger.debug(f"Skipping SharePoint URL: {website}")
                    continue
                    
                if name and website:
                    partners.append({
                        "name": name,
                        "website": website
                    })
                    logger.info(f"Added partner: {name}")
    except Exception as e:
        logger.error(f"Error loading partners from CSV: {e}", exc_info=True)
        return []
        
    logger.info(f"Successfully loaded {len(partners)} partners")
    return partners

def sanitize_filename(name: str) -> str:
    """Sanitize string for use in filenames."""
    # Replace invalid characters with underscore
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def save_results(vendor_name: str, results: Dict):
    """Save analysis results to JSON file with improved error handling."""
    try:
        # Create directories if they don't exist
        base_dir = Path("data/vendor_analysis")
        latest_dir = base_dir / "latest"
        archive_dir = base_dir / "archive"
        error_dir = base_dir / "errors"  # New directory for error cases
        
        for directory in [base_dir, latest_dir, archive_dir, error_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Sanitize vendor name for filename
        safe_name = sanitize_filename(vendor_name)
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_file = latest_dir / f"{safe_name}_latest.json"
        archive_file = archive_dir / f"{safe_name}_{timestamp}.json"
        
        # Check if results contain an error
        if "error" in results and not results.get("category_analysis"):
            # Save to error directory instead
            error_file = error_dir / f"{safe_name}_{timestamp}_error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.warning(f"Saved error results for {vendor_name} to {error_file}")
            return
        
        # Archive existing latest file if it exists
        if latest_file.exists():
            try:
                import shutil
                backup_file = archive_dir / f"{safe_name}_{timestamp}.json"
                shutil.copy(str(latest_file), str(backup_file))
                logger.debug(f"Backed up previous results for {vendor_name}")
            except Exception as e:
                logger.warning(f"Failed to backup previous results for {vendor_name}: {e}")
            
        # Save new results
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Also save a timestamped copy in the archive
        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Update progress tracking file
        update_progress_tracking(vendor_name, "success")
            
        logger.info(f"Saved analysis results for {vendor_name}")
    except Exception as e:
        logger.error(f"Error saving results for {vendor_name}: {e}")
        # Try to save error information
        try:
            error_info = {
                "vendor_name": vendor_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "partial_results": results
            }
            error_file = error_dir / f"{safe_name}_{timestamp}_save_error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved error information to {error_file}")
            
            # Update progress tracking
            update_progress_tracking(vendor_name, "error", str(e))
        except:
            logger.critical(f"Failed to save error information for {vendor_name}")

def update_progress_tracking(vendor_name: str, status: str, error_message: str = None):
    """Update the progress tracking file with vendor processing status."""
    try:
        progress_file = Path("data/vendor_analysis/progress_tracking.json")
        
        # Load existing progress data
        progress_data = {}
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
            except:
                logger.warning("Failed to load existing progress data, creating new file")
        
        # Update with new information
        progress_data[vendor_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        
        # Save updated progress data
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to update progress tracking: {e}")

def get_progress_summary():
    """Generate a summary of processing progress."""
    try:
        progress_file = Path("data/vendor_analysis/progress_tracking.json")
        if not progress_file.exists():
            return "No progress tracking data available."
            
        with open(progress_file, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
            
        total = len(progress_data)
        successful = sum(1 for v in progress_data.values() if v.get("status") == "success")
        errors = sum(1 for v in progress_data.values() if v.get("status") == "error")
        
        return f"Progress: {successful}/{total} vendors processed successfully ({errors} errors)"
    except Exception as e:
        return f"Failed to generate progress summary: {e}"

async def get_technology_categories() -> List[Dict]:
    """Fetch technology categories and their keywords from Neo4j."""
    try:
        neo4j_service = Neo4jService()
        categories = await neo4j_service.get_all_categories()
        logger.info(f"Fetched {len(categories)} technology categories from Neo4j")
        
        # Log category structure for debugging
        if categories:
            logger.debug("First category structure:")
            logger.debug(json.dumps(categories[0], indent=2))
            
        return categories
    except Exception as e:
        logger.error(f"Error fetching technology categories: {str(e)}")
        raise
    finally:
        await neo4j_service.cleanup()

def create_category_analysis_prompt(category: Dict, website_analysis: Dict) -> str:
    """Create an enhanced prompt for analyzing a specific technology category."""
    # Extract structured data if available
    structured_data_text = ""
    if website_analysis.get("structured_data"):
        structured_data_text = "Structured Data:\n"
        for key, value in website_analysis.get("structured_data", {}).items():
            if isinstance(value, str):
                structured_data_text += f"{key}: {value}\n"
    
    # Get content with increased length
    content = website_analysis.get('content', '')[:3500]
    
    return f"""You are an expert AI technology analyst specializing in federal AI use cases. Analyze the vendor's capabilities in the {category['name']} technology category with precision and depth.

CATEGORY DETAILS:
Category: {category['name']}
Definition: {category['category_definition']}

Technical Keywords: {', '.join(category.get('technical_keywords', []))}
Business Terms: {', '.join(category.get('business_terms', []))}
Core Capabilities: {', '.join(category.get('capabilities', []))}

VENDOR INFORMATION:
Website Title: {website_analysis.get('title', '')}
Website Description: {website_analysis.get('description', '')}

CONTENT TO ANALYZE:
{content}

{structured_data_text}

ANALYSIS INSTRUCTIONS:
1. Carefully evaluate how well the vendor's capabilities align with the {category['name']} category
2. Look for explicit mentions of technologies, implementations, and use cases
3. Consider both direct and indirect evidence of capabilities
4. Assess the depth and breadth of their offerings in this category
5. Evaluate technical sophistication and maturity level
6. Consider integration capabilities with other systems

RESPONSE FORMAT:
Provide your analysis in the following JSON structure:

{{
    "confidence_score": float,  // 0.0-1.0 based on evidence strength and alignment
    "summary": string,  // 2-3 sentence executive summary of capabilities in this category
    "evidence": [  // 3-5 most compelling pieces of evidence
        {{
            "text": string,  // Direct quote or paraphrase from source material
            "relevance": float,  // 0.0-1.0 indicating how strongly this supports capability
            "source": string  // Where this evidence was found (e.g., "Website Content", "Product Description")
        }}
    ],
    "capabilities": [string],  // Specific capabilities identified within this category
    "technical_keywords": [string],  // Technical terms found that align with this category
    "business_terms": [string],  // Business/domain terms relevant to this category
    "implementation_details": string,  // How they implement this technology (architecture, approach, etc.)
    "integration_points": [string]  // How this technology integrates with other systems or technologies
}}

QUALITY GUIDELINES:
- Assign confidence scores objectively based on evidence quality and quantity
- Only include capabilities with supporting evidence
- Be specific and precise in your analysis
- Focus on technical capabilities rather than marketing claims
- Identify both strengths and potential limitations
- Only include high-confidence findings with clear evidence
- For implementation_details, provide a paragraph with technical depth when possible

Your analysis will be used to evaluate this vendor's suitability for federal AI projects, so accuracy and thoroughness are essential."""

def parse_category_analysis(response: str) -> Dict:
    """Parse the LLM response into structured category analysis with improved error handling."""
    try:
        # Try multiple approaches to extract JSON
        json_text = None
        
        # Approach 1: Extract from markdown code blocks
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0].strip()
        
        # Approach 2: Find JSON object pattern
        if not json_text:
            import re
            json_pattern = re.search(r'(\{[\s\S]*\})', response)
            if json_pattern:
                json_text = json_pattern.group(1).strip()
        
        # Approach 3: Use the whole response
        if not json_text:
            json_text = response.strip()
        
        # Try to parse the JSON
        try:
            analysis = json.loads(json_text)
        except json.JSONDecodeError as e:
            # Attempt to fix common JSON issues
            logger.warning(f"Initial JSON parsing failed: {e}. Attempting to fix JSON.")
            
            # Fix missing quotes around keys
            fixed_json = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', json_text)
            
            # Fix single quotes to double quotes
            fixed_json = fixed_json.replace("'", '"')
            
            # Fix trailing commas
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*]', ']', fixed_json)
            
            # Try parsing again
            try:
                analysis = json.loads(fixed_json)
            except json.JSONDecodeError:
                # Last resort: extract fields individually using regex
                logger.warning("JSON fixing failed. Attempting to extract fields individually.")
                analysis = extract_fields_with_regex(json_text)
        
        # Ensure required fields are present with correct types
        required_fields = [
            'confidence_score', 
            'summary', 
            'evidence', 
            'capabilities', 
            'technical_keywords',
            'business_terms',
            'implementation_details',
            'integration_points'
        ]
        
        for field in required_fields:
            if field not in analysis:
                if field in ['evidence', 'capabilities', 'technical_keywords', 'business_terms', 'integration_points']:
                    analysis[field] = []
                else:
                    analysis[field] = ""
                    
        # Initialize GitHub evidence if not present
        if 'github_evidence' not in analysis:
            analysis['github_evidence'] = []
                
        # Ensure confidence score is a float between 0 and 1
        try:
            analysis['confidence_score'] = float(analysis['confidence_score'])
            analysis['confidence_score'] = max(0.0, min(1.0, analysis['confidence_score']))
        except (ValueError, TypeError):
            analysis['confidence_score'] = 0.0
            
        # Ensure evidence items have proper structure
        if isinstance(analysis['evidence'], list):
            structured_evidence = []
            for item in analysis['evidence']:
                if isinstance(item, dict):
                    if 'text' not in item:
                        continue
                    
                    # Ensure relevance is a float
                    try:
                        relevance = float(item.get('relevance', 0.5))
                    except (ValueError, TypeError):
                        relevance = 0.5
                        
                    structured_evidence.append({
                        'text': item['text'],
                        'relevance': max(0.0, min(1.0, relevance)),
                        'source': item.get('source', 'Content')
                    })
                elif isinstance(item, str):
                    structured_evidence.append({
                        'text': item,
                        'relevance': 0.5,
                        'source': 'Content'
                    })
            analysis['evidence'] = structured_evidence
            
        # Ensure list fields are actually lists
        list_fields = ['capabilities', 'technical_keywords', 'business_terms', 'integration_points']
        for field in list_fields:
            if not isinstance(analysis[field], list):
                if isinstance(analysis[field], str):
                    # Convert comma-separated string to list
                    analysis[field] = [item.strip() for item in analysis[field].split(',') if item.strip()]
                else:
                    analysis[field] = []
        
        # Ensure implementation_details is a string
        if not isinstance(analysis['implementation_details'], str):
            if isinstance(analysis['implementation_details'], dict):
                # Handle nested structure sometimes returned by LLMs
                paragraphs = []
                if 'paragraphs' in analysis['implementation_details']:
                    for p in analysis['implementation_details']['paragraphs']:
                        if isinstance(p, dict) and 'text' in p:
                            paragraphs.append(p['text'])
                        elif isinstance(p, str):
                            paragraphs.append(p)
                analysis['implementation_details'] = ' '.join(paragraphs)
            else:
                analysis['implementation_details'] = str(analysis['implementation_details'])
            
        return analysis
    except Exception as e:
        logger.error(f"Error parsing category analysis: {str(e)}")
        return {
            "confidence_score": 0.0,
            "summary": "Failed to parse analysis results.",
            "evidence": [],
            "capabilities": [],
            "technical_keywords": [],
            "business_terms": [],
            "implementation_details": "",
            "integration_points": [],
            "github_evidence": []
        }

def extract_fields_with_regex(text: str) -> Dict:
    """Extract fields from text using regex when JSON parsing fails."""
    import re
    
    result = {
        'confidence_score': 0.0,
        'summary': "",
        'evidence': [],
        'capabilities': [],
        'technical_keywords': [],
        'business_terms': [],
        'implementation_details': "",
        'integration_points': [],
        'github_evidence': []
    }
    
    # Extract confidence score
    confidence_match = re.search(r'"confidence_score"\s*:\s*([\d\.]+)', text)
    if confidence_match:
        try:
            result['confidence_score'] = float(confidence_match.group(1))
        except:
            pass
    
    # Extract summary
    summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
    if summary_match:
        result['summary'] = summary_match.group(1)
    
    # Extract implementation details
    impl_match = re.search(r'"implementation_details"\s*:\s*"([^"]+)"', text)
    if impl_match:
        result['implementation_details'] = impl_match.group(1)
    
    # Extract list items
    for field in ['capabilities', 'technical_keywords', 'business_terms', 'integration_points']:
        items = re.findall(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if items:
            # Extract quoted strings from the list
            matches = re.findall(r'"([^"]+)"', items[0])
            result[field] = matches
    
    return result

async def analyze_vendor(vendor_name: str, website_url: str, ollama_url: str, model: str) -> Dict[str, Any]:
    """Analyze a vendor's capabilities."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis for vendor: {vendor_name}")
    logger.info(f"Analyzing website: {website_url}")
    
    try:
        # Ensure vendor_name is not None
        if vendor_name is None:
            vendor_name = ""
            logger.warning("Vendor name is None, using empty string")
            
        # Initialize Ollama client
        async with OllamaClient(ollama_url, model) as ollama:
            if not await ollama.check_connection():
                logger.error("Failed to establish connection with Ollama server")
                return {
                    "vendor_name": vendor_name,
                    "error": "Failed to connect to Ollama server",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get technology categories from database
            categories = await get_technology_categories()
            if not categories:
                logger.error("Failed to fetch technology categories from database")
                return {
                    "vendor_name": vendor_name,
                    "error": "Failed to fetch technology categories",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Website analysis
            analyzer = WebsiteAnalyzer()
            try:
                website_analysis = await analyzer.analyze_website(website_url)
                logger.info("Website analysis completed")
                logger.debug(f"Website analysis results: {website_analysis}")
            finally:
                await analyzer.cleanup()
            
            # Initialize searcher for GitHub and web results
            searcher = MultiSourceSearcher(vendor_name, ollama)
            await searcher.initialize()
            
            # Category analysis
            logger.info("Starting category analysis")
            category_results = {}
            
            for category in categories:
                logger.debug(f"Analyzing category: {category['name']}")
                try:
                    # Ensure category has a name
                    category_name = category.get('name', '')
                    if not category_name:
                        logger.warning("Category missing name, skipping")
                        continue
                        
                    # Get search results including GitHub
                    search_results = await searcher.search_for_category(category, website_analysis)
                    
                    # Safely construct GitHub search query
                    github_query = f"{vendor_name} {category_name}"
                    logger.debug(f"GitHub search query: {github_query}")
                    github_results = await searcher.search_github(github_query)
                    
                    # Create analysis prompt
                    prompt = create_category_analysis_prompt(category, website_analysis)
                    response = await ollama.generate(prompt)
                    
                    if response is None:
                        logger.error(f"Failed to get response for category: {category_name}")
                        continue
                    
                    # Parse analysis results
                    analysis = parse_category_analysis(response)
                    
                    # Add GitHub evidence if relevant
                    if github_results:
                        analysis['github_evidence'] = [
                            {
                                'repo_name': result['title'],
                                'description': result['description'],
                                'stars': result['stars'],
                                'url': result['link']
                            }
                            for result in github_results
                        ]
                        
                        # Adjust confidence score based on GitHub evidence
                        if analysis['github_evidence']:
                            github_confidence = min(len(analysis['github_evidence']) * 0.1, 0.3)  # Max 0.3 boost from GitHub
                            analysis['confidence_score'] = min(1.0, analysis['confidence_score'] + github_confidence)
                    else:
                        # Ensure github_evidence is always initialized
                        analysis['github_evidence'] = []
                    
                    # Only include categories with sufficient confidence
                    if analysis['confidence_score'] >= DEFAULT_CONFIDENCE_THRESHOLD:
                        category_results[category_name] = analysis
                    
                except Exception as e:
                    logger.error(f"Error analyzing category {category.get('name', 'unknown')}: {str(e)}")
                    continue
            
            await searcher.cleanup()
            
            return {
                "vendor_name": vendor_name,
                "website": website_url,
                "website_analysis": website_analysis,
                "category_analysis": category_results,
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "3.0"
            }
            
    except Exception as e:
        logger.error(f"Error during vendor analysis: {str(e)}")
        return {
            "vendor_name": vendor_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def main():
    start_time = datetime.now()
    
    # Set global confidence threshold
    global DEFAULT_CONFIDENCE_THRESHOLD
    
    parser = argparse.ArgumentParser(description="Enhanced vendor analysis script")
    parser.add_argument("--limit", type=int, help="Limit number of partners to process")
    parser.add_argument("--csv-file", default="data/input/Dell Federal AI Partner Tracking.csv",
                      help="Path to CSV file containing partner data")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                      help="Minimum confidence score for including category analysis")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                      help="Ollama API URL")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL,
                      help="Ollama model to use")
    parser.add_argument("--parallel", type=int, default=1,
                      help="Number of partners to process in parallel (default: 1)")
    parser.add_argument("--resume", action="store_true",
                      help="Resume processing from where it left off by skipping already analyzed partners")
    parser.add_argument("--vendor", type=str,
                      help="Process a specific vendor by name")
    parser.add_argument("--summary-only", action="store_true",
                      help="Only display progress summary without processing any vendors")
    args = parser.parse_args()
    
    # Update confidence threshold from args
    DEFAULT_CONFIDENCE_THRESHOLD = args.confidence_threshold
    
    logger.info("Starting vendor analysis script")
    logger.info(f"Arguments: limit={args.limit}, csv_file={args.csv_file}, confidence_threshold={args.confidence_threshold}, "
                f"ollama_url={args.ollama_url}, model={args.model}, parallel={args.parallel}, resume={args.resume}")
    
    # Display current progress
    progress_summary = get_progress_summary()
    logger.info(f"Current progress: {progress_summary}")
    
    # If summary-only flag is set, just display progress and exit
    if args.summary_only:
        logger.info("Summary-only mode: exiting without processing vendors")
        return
    
    # Create a summary file with execution details
    execution_summary = {
        "start_time": start_time.isoformat(),
        "arguments": vars(args),
        "initial_progress": progress_summary
    }
    
    try:
        # Load partners from CSV
        partners = load_partners(args.csv_file, args.limit)
        if not partners:
            logger.error("No partners loaded from CSV")
            return
        
        logger.info(f"Loaded {len(partners)} partners")
        
        # Filter for specific vendor if requested
        if args.vendor:
            partners = [p for p in partners if p["name"].lower() == args.vendor.lower()]
            if not partners:
                logger.error(f"Vendor '{args.vendor}' not found in CSV file")
                return
            logger.info(f"Filtered to process only vendor: {args.vendor}")
        
        # Skip already processed partners if resuming
        if args.resume:
            # Get list of already processed partners
            processed_partners = set()
            latest_dir = Path("data/vendor_analysis/latest")
            if latest_dir.exists():
                for file in latest_dir.glob("*_latest.json"):
                    vendor_name = file.name.replace("_latest.json", "")
                    processed_partners.add(vendor_name)
            
            # Filter out already processed partners
            original_count = len(partners)
            partners = [p for p in partners if sanitize_filename(p["name"]) not in processed_partners]
            logger.info(f"Resuming: Skipping {original_count - len(partners)} already processed partners")
        
        # Process partners in parallel or sequentially
        if args.parallel > 1:
            logger.info(f"Processing {len(partners)} partners with parallelism of {args.parallel}")
            
            # Create batches of partners
            batches = [partners[i:i + args.parallel] for i in range(0, len(partners), args.parallel)]
            
            for batch_num, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_num + 1}/{len(batches)} with {len(batch)} partners")
                
                # Process batch in parallel
                tasks = []
                for partner in batch:
                    task = asyncio.create_task(process_partner(
                        partner["name"],
                        partner["website"],
                        args.ollama_url,
                        args.model
                    ))
                    tasks.append(task)
                
                # Wait for all tasks in batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for partner, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing partner {partner['name']}: {result}")
                    else:
                        save_results(partner["name"], result)
                        logger.info(f"Completed analysis for {partner['name']}")
        else:
            # Process sequentially
            logger.info(f"Processing {len(partners)} partners sequentially")
            for partner in partners:
                try:
                    logger.info(f"Processing partner: {partner['name']}")
                    results = await analyze_vendor(
                        partner["name"], 
                        partner["website"],
                        args.ollama_url,
                        args.model
                    )
                    save_results(partner["name"], results)
                    logger.info(f"Completed analysis for {partner['name']}")
                except Exception as e:
                    logger.error(f"Error processing partner {partner['name']}: {e}", exc_info=True)
                    continue
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Update and display final progress
        final_progress = get_progress_summary()
        logger.info(f"Final progress: {final_progress}")
        logger.info(f"Total execution time: {execution_time}")
        
        # Save execution summary
        execution_summary.update({
            "end_time": end_time.isoformat(),
            "execution_time_seconds": execution_time.total_seconds(),
            "final_progress": final_progress
        })
        
        summary_dir = Path("data/vendor_analysis/execution_summaries")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / f"execution_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(execution_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Execution summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save execution summary: {e}")
        
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        # Try to save error information
        try:
            error_info = {
                "vendor_name": "Unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "execution_summary": execution_summary
            }
            error_file = summary_dir / f"execution_summary_{start_time.strftime('%Y%m%d_%H%M%S')}_error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved error information to {error_file}")
            
            # Update progress tracking
            update_progress_tracking("Unknown", "error", str(e))
        except:
            logger.critical(f"Failed to save error information")

async def process_partner(name: str, website: str, ollama_url: str, model: str) -> Dict:
    """Process a single partner for parallel execution."""
    try:
        logger.info(f"Starting analysis for partner: {name}")
        results = await analyze_vendor(name, website, ollama_url, model)
        logger.info(f"Analysis completed for partner: {name}")
        return results
    except Exception as e:
        logger.error(f"Error in process_partner for {name}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 