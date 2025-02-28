#!/usr/bin/env python3
"""
Partner Analysis Script

This script analyzes partners to determine which AI technology categories they match with.
It uses a combination of bootstrap information, website content, and search results to
generate an analysis of the partner's capabilities.

Usage:
    python partner_analysis.py [options]

Options:
    --limit N                  Limit the number of partners to process
    --csv-file FILE            Path to the CSV file containing partner data
    --categories-file FILE     Path to the CSV file containing AI technology categories
    --output-dir DIR           Directory to save output files
    --output-format FORMAT     Output format (json or csv)
    --ollama-url URL           URL for Ollama API
    --ollama-model MODEL       Model to use for Ollama analysis
    --bootstrap-model MODEL    Model to use for bootstrap information
    --openai-model MODEL       Model to use for OpenAI
    --openai-primary           Use OpenAI as primary model
    --openai-fallback          Use OpenAI as fallback model
    --skip-website             Skip fetching website content
    --skip-search              Skip Brave search enhancement
    --skip-github              Skip GitHub search enhancement
    --verbose                  Enable verbose logging
    --debug                    Enable debug logging
"""

import os
import re
import csv
import json
import time
import logging
import asyncio
import argparse
import datetime
import traceback
import urllib.parse
from typing import Dict, List, Any, Optional, Union

import aiohttp
import openai
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables from .env file
load_dotenv()

# Default models
DEFAULT_BOOTSTRAP_MODEL = "llama3:8b-instruct-fp16"
DEFAULT_ANALYSIS_MODEL = "deepseek-r1:70b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Default URLs
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Default paths
DEFAULT_OUTPUT_DIR = "data/output/partner_analysis"
DEFAULT_CATEGORIES_FILE = "data/input/AI-Technology-Categories-v1.4.csv"
DEFAULT_PARTNERS_FILE = "data/input/partners.csv"

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process partners and generate analysis results.")
    
    # Input/output options
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of partners to process")
    parser.add_argument("--csv-file", type=str, default="data/input/partners.csv", help="Path to the CSV file containing partner data")
    parser.add_argument("--categories-file", type=str, default="data/input/AI-Technology-Categories-v1.4.csv", help="Path to the CSV file containing AI technology categories")
    parser.add_argument("--output-dir", type=str, default="data/output/partner_analysis", help="Directory to save output files")
    parser.add_argument("--output-format", type=str, default="json", choices=["json", "csv", "both"], help="Output format (json, csv, or both)")
    
    # Model options
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL for Ollama API")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_ANALYSIS_MODEL, help="Model to use for Ollama analysis")
    parser.add_argument("--bootstrap-model", type=str, default=DEFAULT_BOOTSTRAP_MODEL, help="Model to use for bootstrap information")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL, help="Model to use for OpenAI")
    parser.add_argument("--openai-primary", action="store_true", help="Use OpenAI as primary model")
    parser.add_argument("--openai-fallback", action="store_true", help="Use OpenAI as fallback model")
    
    # Feature flags
    parser.add_argument("--skip-website", action="store_true", help="Skip fetching website content")
    parser.add_argument("--skip-search", action="store_true", help="Skip Brave search enhancement")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub search enhancement")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("partner_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def _load_ai_categories(categories_file):
    """Load AI technology categories from a CSV file."""
    print(f"Loading AI technology categories from: {categories_file}")
    
    categories = []
    
    try:
        with open(categories_file, "r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names to ensure consistency
                cleaned_row = {}
                for key, value in row.items():
                    # Remove BOM character if present
                    clean_key = key.replace('\ufeff', '')
                    cleaned_row[clean_key] = value
                
                # Extract category information
                category = {
                    "name": cleaned_row.get("ai_category", "").strip(),
                    "description": cleaned_row.get("definition", "").strip(),
                    "keywords": cleaned_row.get("keywords", "").strip(),
                    "capabilities": cleaned_row.get("capabilities", "").strip(),
                    "business_language": cleaned_row.get("business_language", "").strip(),
                    "maturity_level": cleaned_row.get("maturity_level", "").strip(),
                    "zone": cleaned_row.get("zone", "").strip()
                }
                
                if category["name"]:
                    categories.append(category)
    except Exception as e:
        print(f"Error loading AI technology categories: {str(e)}")
        return []
    
    print(f"Loaded {len(categories)} AI technology categories")
    return categories

def _load_partners_from_csv(csv_file, limit=None):
    """Load partners from a CSV file."""
    print(f"Loading partners from CSV file: {csv_file}")
    
    partners = []
    
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                partner_name = row.get("partner_name", "").strip()
                if partner_name:
                    partners.append(row)
                    if limit and len(partners) >= limit:
                        break
    except Exception as e:
        print(f"Error loading partners from CSV file: {str(e)}")
        return []
    
    print(f"Loaded {len(partners)} partners from CSV file")
    return partners

async def _get_website_content(url):
    """Get website content using Playwright for JavaScript rendering."""
    print(f"Fetching website content from: {url}")
    
    try:
        # Initialize Playwright
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            
            # Create a new page
            page = await browser.new_page()
            
            # Set timeout for navigation
            page.set_default_timeout(30000)  # 30 seconds
            
            # Navigate to the URL
            try:
                await page.goto(url, wait_until="networkidle")
            except Exception as e:
                print(f"Error navigating to {url}: {str(e)}")
                await browser.close()
                return None
            
            # Wait for the page to load
            await asyncio.sleep(2)
            
            # Get the page title
            title = await page.title()
            
            # Get the page description
            description = await page.evaluate("""
                () => {
                    const metaDescription = document.querySelector('meta[name="description"]');
                    return metaDescription ? metaDescription.getAttribute('content') : '';
                }
            """)
            
            # Get the page content
            content = await page.evaluate("""
                () => {
                    // Remove scripts, styles, and other non-content elements
                    const scripts = document.querySelectorAll('script, style, noscript, iframe, img, svg, canvas, video, audio');
                    scripts.forEach(s => s.remove());
                    
                    // Get the main content
                    const body = document.body;
                    return body.innerText;
                }
            """)
            
            # Clean up the content
            clean_content = re.sub(r'\s+', ' ', content).strip()
            
            # Close the browser
            await browser.close()
            
            # Return the website content
            return {
                "title": title,
                "description": description,
                "content": clean_content,
                "url": url
            }
    except Exception as e:
        print(f"Error fetching website content: {str(e)}")
        return None

async def _get_brave_search_results(query, max_retries=3):
    """Get search results from Brave Search API."""
    print(f"Getting search results for: {query}")
    
    # Check if Brave Search API key is available
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("Brave API key not found in environment variables")
        return []
    
    # Construct the API URL
    api_url = "https://api.search.brave.com/res/v1/web/search"
    
    # Set up the headers
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_api_key
    }
    
    # Set up the parameters
    params = {
        "q": query,
        "count": 10,
        "search_lang": "en"
    }
    
    # Initialize retry counter
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        # Parse the response
                        data = await response.json()
                        
                        # Extract the search results
                        results = []
                        for web in data.get("web", {}).get("results", []):
                            results.append({
                                "title": web.get("title", ""),
                                "description": web.get("description", ""),
                                "url": web.get("url", ""),
                                "source": "brave_search"
                            })
                        
                        print(f"Found {len(results)} search results for {query}")
                        return results
                    elif response.status == 429:
                        # Rate limiting, wait and retry
                        retry_count += 1
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"Brave Search API rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Other error
                        print(f"Brave Search API returned status code {response.status}")
                        return []
        except Exception as e:
            print(f"Error getting search results: {str(e)}")
            return []
    
    print(f"Failed to get search results after {max_retries} retries")
    return []

async def _search_github(vendor_name, max_retries=3):
    """
    Search GitHub for repositories related to the vendor.
    
    Args:
        vendor_name (str): The name of the vendor to search for
        max_retries (int): Maximum number of retry attempts for rate limiting
        
    Returns:
        list: A list of search results
    """
    if not vendor_name:
        logging.warning("Vendor name is None, skipping GitHub search")
        return []
    
    print(f"Searching GitHub for: {vendor_name}")
    
    # Construct multiple search queries for better coverage
    search_queries = [
        f"{vendor_name} in:name",  # Repositories with vendor name in repo name
        f"{vendor_name} in:description",  # Repositories with vendor name in description
        f"{vendor_name} in:readme",  # Repositories with vendor name in readme
        f"{vendor_name} in:topics",  # Repositories with vendor name in topics
    ]
    
    all_results = []
    
    for query in search_queries:
        if not query:
            continue
            
        retry_count = 0
        while retry_count < max_retries:
            try:
                url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page=10"
                headers = {"Accept": "application/vnd.github.v3+json"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("items", [])
                            
                            # Extract relevant information
                            for repo in results:
                                all_results.append({
                                    "name": repo.get("name"),
                                    "full_name": repo.get("full_name"),
                                    "description": repo.get("description"),
                                    "url": repo.get("html_url"),
                                    "stars": repo.get("stargazers_count"),
                                    "forks": repo.get("forks_count"),
                                    "language": repo.get("language"),
                                    "topics": repo.get("topics", []),
                                    "source": "github"
                                })
                            
                            # Break out of retry loop on success
                            break
                        elif response.status == 403:
                            # Rate limiting, wait and retry
                            retry_count += 1
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logging.warning(f"GitHub API rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                            await asyncio.sleep(wait_time)
                        else:
                            logging.warning(f"GitHub API returned status code {response.status}")
                            break
            except Exception as e:
                logging.error(f"Error searching GitHub: {str(e)}")
                break
    
    print(f"Found {len(all_results)} GitHub results for {vendor_name}")
    return all_results

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model_name=DEFAULT_BOOTSTRAP_MODEL, host="http://localhost:11434"):
        """Initialize the Ollama client."""
        self.model_name = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        print(f"Initialized Ollama client with model: {model_name}")
    
    async def generate(self, prompt):
        """Generate a response from Ollama."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Ollama API returned status code {response.status}: {error_text}")
                        raise Exception(f"Ollama API returned status code {response.status}")
                    
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            print(f"Error generating response from Ollama: {str(e)}")
            raise

class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        # Set up OpenAI API key
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        print(f"INITIALIZED OPENAI CLIENT: {model_name}")
    
    async def generate(self, prompt):
        """Generate a response from OpenAI."""
        print(f"GENERATING RESPONSE FROM OPENAI: {self.model_name}")
    
        # Set up the request
        try:
            # Log the request
            logging.info(f"Sending generation request to OpenAI (attempt 1/3)")
            logging.debug(f"Prompt length: {len(prompt)} characters")
            
            # Make the request
            response = await openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert technology analyst specializing in AI capabilities assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )
            
            # Log the response
            logging.info(f"Received response from OpenAI (length: {len(response.choices[0].message.content)} characters)")
            logging.debug(f"Response preview: {response.choices[0].message.content[:100]}...")
            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {str(e)}")
            print(f"ERROR GENERATING RESPONSE FROM OPENAI: {str(e)}")
            return None

def parse_llm_json_response(response_text):
    """
    Parse JSON from LLM response text with multiple fallback mechanisms.
    
    This function implements several strategies to extract valid JSON from LLM responses:
    1. Try to parse the entire response as JSON
    2. Look for JSON between triple backticks
    3. Look for JSON between single backticks
    4. Look for JSON between curly braces
    5. Try to fix common JSON formatting issues and retry
    
    Args:
        response_text (str): The raw text response from an LLM
        
    Returns:
        dict: The parsed JSON object or None if parsing fails
    """
    if not response_text:
        logging.error("Empty response received from LLM")
        return None
    
    # Print the raw response for debugging
    print("Raw LLM response:")
    print(response_text)
    print("End of raw response")
    
    # Strategy 1: Try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logging.debug(f"Could not parse entire response as JSON: {str(e)}")
    
    # Strategy 2: Look for JSON between triple backticks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, response_text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError as e:
            logging.debug(f"Could not parse JSON between triple backticks: {str(e)}")
    
    # Strategy 3: Look for JSON between single backticks
    json_pattern = r"`([\s\S]*?)`"
    matches = re.findall(json_pattern, response_text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError as e:
            logging.debug(f"Could not parse JSON between single backticks: {str(e)}")
    
    # Strategy 4: Look for JSON between curly braces (outermost pair)
    try:
        # Find the first opening brace and the last closing brace
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = response_text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.debug(f"Could not parse JSON between curly braces: {str(e)}")
    
    # Strategy 5: Try to fix common JSON formatting issues and retry
    try:
        # Replace single quotes with double quotes (common LLM mistake)
        fixed_text = re.sub(r"'([^']*)':\s*'([^']*)'", r'"\1": "\2"', response_text)
        fixed_text = re.sub(r"'([^']*)':\s*\[", r'"\1": [', fixed_text)
        fixed_text = re.sub(r"'([^']*)'", r'"\1"', fixed_text)
        
        # Find JSON-like structure
        start_idx = fixed_text.find('{')
        end_idx = fixed_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = fixed_text[start_idx:end_idx+1]
            return json.loads(json_str)
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"All JSON parsing strategies failed: {str(e)}")
    
    logging.error("Failed to parse JSON from LLM response")
    return None

def _create_bootstrap_prompt(partner):
    """Create a bootstrap prompt for a partner."""
    print(f"Creating bootstrap prompt for {partner}")
    
    prompt = f"""
You are an expert technology analyst specializing in AI capabilities assessment. Your task is to gather basic information about the partner "{partner}" to help with further analysis.

Please provide the following information about {partner} in JSON format:
{{
  "partner_name": "{partner}",
  "primary_business": "Brief description of their primary business focus",
  "key_technologies": ["List of key AI technologies they use or develop"],
  "key_capabilities": ["List of key AI capabilities they offer"],
  "target_industries": ["List of industries they primarily serve"],
  "notable_products": ["List of their notable AI products or solutions"],
  "summary": "Brief summary of their AI capabilities"
}}

Please provide your response as a single, valid JSON object.
"""
    
    return prompt

def _create_analysis_prompt(partner, bootstrap_info, website_content, ai_categories, search_results=None, github_results=None):
    """Create an analysis prompt for a partner."""
    print(f"Creating analysis prompt for {partner}")
    
    # Format the AI categories
    categories_text = "\n".join([f"{i+1}. {cat['name']}: {cat['description']}" for i, cat in enumerate(ai_categories)])
    
    # Format the bootstrap information
    bootstrap_text = json.dumps(bootstrap_info, indent=2) if bootstrap_info else "No bootstrap information available."
    
    # Format the website content
    website_text = ""
    if website_content:
        website_text = f"""
WEBSITE TITLE: {website_content.get('title', 'N/A')}
WEBSITE DESCRIPTION: {website_content.get('description', 'N/A')}
WEBSITE CONTENT: {website_content.get('content', 'N/A')[:5000]}
"""
    else:
        website_text = "No website content available."
    
    # Format search results
    search_text = ""
    if search_results and len(search_results) > 0:
        search_text = "SEARCH RESULTS:\n"
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
            search_text += f"""
Result {i+1}:
Title: {result.get('title', 'N/A')}
Description: {result.get('description', 'N/A')}
URL: {result.get('url', 'N/A')}
"""
    
    # Format GitHub results
    github_text = ""
    if github_results and len(github_results) > 0:
        github_text = "GITHUB REPOSITORIES:\n"
        for i, repo in enumerate(github_results[:5]):  # Limit to top 5 repos
            github_text += f"""
Repository {i+1}:
Name: {repo.get('name', 'N/A')}
Description: {repo.get('description', 'N/A')}
URL: {repo.get('url', 'N/A')}
Stars: {repo.get('stars', 'N/A')}
Language: {repo.get('language', 'N/A')}
Topics: {', '.join(repo.get('topics', []))}
"""
    
    # Create the analysis prompt
    prompt = f"""
You are an expert technology analyst specializing in AI capabilities assessment. Your task is to analyze the partner "{partner}" and determine which AI technology categories they match with based on their capabilities.

Here is the information about the partner:

BOOTSTRAP INFORMATION:
{bootstrap_text}

WEBSITE INFORMATION:
{website_text}
{search_text}
{github_text}

Here are the AI technology categories to consider:
{categories_text}

Based on the information provided, analyze which AI technology categories the partner matches with. For each matching category, provide:
1. A confidence score (0.0 to 1.0) indicating how strongly the partner matches with the category
2. Evidence from the provided information that supports the match

Return your analysis in the following JSON format:
{{
  "partner_name": "{partner}",
  "matched_categories": [
    {{
      "category": "Category Name",
      "confidence": 0.95,
      "evidence": "Specific evidence from the provided information that supports this match"
    }},
    ...
  ]
}}

Only include categories where you have found evidence of a match. Rank the matched categories by confidence score in descending order.
"""
    
    return prompt

async def _get_bootstrap_info_ollama(partner, model_name=DEFAULT_BOOTSTRAP_MODEL):
    """Get bootstrap information for a partner using Ollama."""
    print(f"Getting bootstrap information for {partner} using Ollama")
    
    # Create the bootstrap prompt
    bootstrap_prompt = _create_bootstrap_prompt(partner)
    
    # Initialize the Ollama client
    ollama_client = OllamaClient(model_name=model_name)
    
    try:
        # Generate the bootstrap response
        bootstrap_response = await ollama_client.generate(bootstrap_prompt)
        print(f"Received bootstrap response for {partner} (length: {len(bootstrap_response)} characters)")
        
        # Parse the bootstrap response
        bootstrap_info = parse_llm_json_response(bootstrap_response)
        
        if bootstrap_info:
            print(f"Successfully parsed bootstrap information for {partner}")
            return bootstrap_info
        else:
            print(f"Failed to parse bootstrap information for {partner}")
            return None
    except Exception as e:
        print(f"Error getting bootstrap information for {partner}: {str(e)}")
        return None

async def _get_bootstrap_info_openai(partner, model_name=DEFAULT_OPENAI_MODEL):
    """Get bootstrap information for a partner using OpenAI."""
    print(f"Getting bootstrap information for {partner} using OpenAI")
    
    # Create the bootstrap prompt
    bootstrap_prompt = _create_bootstrap_prompt(partner)
    
    # Initialize the OpenAI client
    openai_client = OpenAIClient(model_name=model_name)
    
    try:
        # Generate the bootstrap response
        bootstrap_response = await openai_client.generate(bootstrap_prompt)
        print(f"Received bootstrap response for {partner} (length: {len(bootstrap_response)} characters)")
        
        # Parse the bootstrap response
        bootstrap_info = parse_llm_json_response(bootstrap_response)
        
        if bootstrap_info:
            print(f"Successfully parsed bootstrap information for {partner}")
            return bootstrap_info
        else:
            print(f"Failed to parse bootstrap information for {partner}")
            return None
    except Exception as e:
        print(f"Error getting bootstrap information for {partner}: {str(e)}")
        return None

async def _get_bootstrap_info(partner, use_openai=False):
    """Get bootstrap information for a partner."""
    if use_openai:
        return await _get_bootstrap_info_openai(partner)
    else:
        return await _get_bootstrap_info_ollama(partner)

async def _get_analysis_ollama(partner, bootstrap_info, website_content, ai_categories, search_results=None, github_results=None, model_name=DEFAULT_ANALYSIS_MODEL):
    """Get analysis for a partner using Ollama."""
    print(f"Getting analysis for {partner} using Ollama")
    
    # Create the analysis prompt
    analysis_prompt = _create_analysis_prompt(partner, bootstrap_info, website_content, ai_categories, search_results, github_results)
    
    # Initialize the Ollama client
    ollama_client = OllamaClient(model_name=model_name)
    
    try:
        # Generate the analysis response
        analysis_response = await ollama_client.generate(analysis_prompt)
        print(f"Received analysis response for {partner} (length: {len(analysis_response)} characters)")
        
        # Parse the analysis response
        analysis_result = parse_llm_json_response(analysis_response)
        
        if analysis_result:
            print(f"Successfully parsed analysis for {partner}")
            return analysis_result
        else:
            print(f"Failed to parse analysis for {partner}")
            return None
    except Exception as e:
        print(f"Error getting analysis for {partner}: {str(e)}")
        return None

async def _get_analysis_openai(partner, bootstrap_info, website_content, ai_categories, search_results=None, github_results=None, model_name=DEFAULT_OPENAI_MODEL):
    """Get analysis for a partner using OpenAI."""
    print(f"Getting analysis for {partner} using OpenAI")
    
    # Create the analysis prompt
    analysis_prompt = _create_analysis_prompt(partner, bootstrap_info, website_content, ai_categories, search_results, github_results)
    
    # Initialize the OpenAI client
    openai_client = OpenAIClient(model_name=model_name)
    
    try:
        # Generate the analysis response
        analysis_response = await openai_client.generate(analysis_prompt)
        print(f"Received analysis response for {partner} (length: {len(analysis_response)} characters)")
        
        # Parse the analysis response
        analysis_result = parse_llm_json_response(analysis_response)
        
        if analysis_result:
            print(f"Successfully parsed analysis for {partner}")
            return analysis_result
        else:
            print(f"Failed to parse analysis for {partner}")
            return None
    except Exception as e:
        print(f"Error getting analysis for {partner}: {str(e)}")
        return None

async def _get_analysis(partner, bootstrap_info, website_content, ai_categories, search_results=None, github_results=None, use_openai=False):
    """Get analysis for a partner."""
    if use_openai:
        return await _get_analysis_openai(partner, bootstrap_info, website_content, ai_categories, search_results, github_results)
    else:
        return await _get_analysis_ollama(partner, bootstrap_info, website_content, ai_categories, search_results, github_results)

async def main():
    """Main function."""
    print("MAIN FUNCTION STARTED")
    
    # Parse command-line arguments
    args = _parse_args()
    print(f"ARGS: {args}")
    
    # Load AI technology categories
    categories = _load_ai_categories(args.categories_file)
    print(f"LOADED {len(categories)} AI TECHNOLOGY CATEGORIES")
    
    # Load partners from CSV
    partners = _load_partners_from_csv(args.csv_file, limit=args.limit)
    print(f"LOADED {len(partners)} PARTNERS")
    
    # Create a list to store all analysis results for consolidated CSV
    all_analysis_results = []
    
    # Process each partner
    for partner in partners:
        partner_name = partner.get("partner_name", "").strip()
        if not partner_name:
            print("SKIPPING PARTNER WITH NO NAME")
            continue
            
        print(f"PROCESSING PARTNER: {partner_name}")
        
        # Get website content if available
        website_content = None
        if not args.skip_website:
            website_url = partner.get("website_url", f"https://{partner_name}")
            print(f"GETTING WEBSITE CONTENT FOR: {website_url}")
            website_content = await _get_website_content(website_url)
        
        # Get bootstrap information
        bootstrap_info = await _get_bootstrap_info(partner_name, args.openai_primary)
        if bootstrap_info:
            print(f"BOOTSTRAP INFO OBTAINED FOR: {partner_name}")
        else:
            print(f"FAILED TO GET BOOTSTRAP INFO FOR: {partner_name}")
        
        # Get search results if available
        search_results = []
        if not args.skip_search:
            print(f"GETTING SEARCH RESULTS FOR: {partner_name}")
            search_results = await _get_brave_search_results(partner_name, max_retries=3)
            print(f"OBTAINED {len(search_results)} SEARCH RESULTS FOR: {partner_name}")
        
        # Get GitHub search results if available
        github_results = []
        if not args.skip_github:
            print(f"GETTING GITHUB RESULTS FOR: {partner_name}")
            github_results = await _search_github(partner_name)
            print(f"OBTAINED {len(github_results)} GITHUB RESULTS FOR: {partner_name}")
        
        # Get analysis
        analysis = None
        
        # Try Ollama first unless OpenAI is primary
        if not args.openai_primary:
            print(f"GETTING OLLAMA ANALYSIS FOR: {partner_name}")
            analysis = await _get_analysis_ollama(partner_name, bootstrap_info, website_content, categories, search_results, github_results, model_name=args.ollama_model)
        
        # Fall back to OpenAI if Ollama fails or OpenAI is primary
        if analysis is None and (args.openai_fallback or args.openai_primary):
            print(f"GETTING OPENAI ANALYSIS FOR: {partner_name}")
            analysis = await _get_analysis_openai(partner_name, bootstrap_info, website_content, categories, search_results, github_results, model_name=args.openai_model)
        
        # Save analysis results
        if analysis:
            print(f"ANALYSIS OBTAINED FOR: {partner_name}")
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual JSON file (always save JSON for reference)
            output_file = os.path.join(args.output_dir, f"{partner_name}_{timestamp}.json")
            
            # Enhance the analysis JSON with additional bootstrap information
            enhanced_analysis = analysis.copy()
            if bootstrap_info:
                # Add bootstrap information to the analysis
                enhanced_analysis["primary_business"] = bootstrap_info.get("primary_business", "")
                enhanced_analysis["key_technologies"] = bootstrap_info.get("key_technologies", [])
                enhanced_analysis["key_capabilities"] = bootstrap_info.get("key_capabilities", [])
                enhanced_analysis["target_industries"] = bootstrap_info.get("target_industries", [])
                enhanced_analysis["notable_products"] = bootstrap_info.get("notable_products", [])
                enhanced_analysis["summary"] = bootstrap_info.get("summary", "")
            
            # Add metadata about the analysis
            enhanced_analysis["analysis_metadata"] = {
                "timestamp": timestamp,
                "analysis_model": args.ollama_model if not args.openai_primary else args.openai_model,
                "bootstrap_model": args.bootstrap_model,
                "data_sources": []
            }
            
            # Add data sources used
            if website_content:
                enhanced_analysis["analysis_metadata"]["data_sources"].append("website")
            if search_results and len(search_results) > 0:
                enhanced_analysis["analysis_metadata"]["data_sources"].append("search")
                enhanced_analysis["analysis_metadata"]["search_results_count"] = len(search_results)
            if github_results and len(github_results) > 0:
                enhanced_analysis["analysis_metadata"]["data_sources"].append("github")
                enhanced_analysis["analysis_metadata"]["github_repos_count"] = len(github_results)
                
                # Add top GitHub repo info if available
                if len(github_results) > 0:
                    sorted_repos = sorted(github_results, key=lambda x: x.get("stars", 0), reverse=True)
                    if sorted_repos:
                        enhanced_analysis["analysis_metadata"]["top_github_repo"] = sorted_repos[0].get("full_name", "")
                        enhanced_analysis["analysis_metadata"]["top_github_stars"] = sorted_repos[0].get("stars", 0)
            
            with open(output_file, "w") as f:
                json.dump(enhanced_analysis, f, indent=2)
            print(f"SAVED ANALYSIS RESULTS TO: {output_file}")
            
            # Store analysis results for consolidated CSV
            for category in analysis.get("matched_categories", []):
                # Extract additional information from bootstrap_info if available
                primary_business = ""
                key_technologies = []
                key_capabilities = []
                target_industries = []
                summary = ""
                notable_products = []
                
                if bootstrap_info:
                    primary_business = bootstrap_info.get("primary_business", "")
                    key_technologies = bootstrap_info.get("key_technologies", [])
                    key_capabilities = bootstrap_info.get("key_capabilities", [])
                    target_industries = bootstrap_info.get("target_industries", [])
                    summary = bootstrap_info.get("summary", "")
                    notable_products = bootstrap_info.get("notable_products", [])
                
                # Search and GitHub metrics
                search_results_count = len(search_results) if search_results else 0
                github_repos_count = len(github_results) if github_results else 0
                
                # Find top GitHub repo by stars
                top_github_repo = ""
                top_github_stars = 0
                if github_results and len(github_results) > 0:
                    # Sort by stars (descending)
                    sorted_repos = sorted(github_results, key=lambda x: x.get("stars", 0), reverse=True)
                    if sorted_repos:
                        top_github_repo = sorted_repos[0].get("full_name", "")
                        top_github_stars = sorted_repos[0].get("stars", 0)
                
                # Analysis metadata
                analysis_model = args.ollama_model if not args.openai_primary else args.openai_model
                bootstrap_model = args.bootstrap_model
                
                # Data sources used
                data_sources = []
                if website_content:
                    data_sources.append("website")
                if search_results and len(search_results) > 0:
                    data_sources.append("search")
                if github_results and len(github_results) > 0:
                    data_sources.append("github")
                
                # Add to consolidated results
                result_entry = {
                    "partner_name": partner_name,
                    "category": category.get("category", ""),
                    "confidence": category.get("confidence", 0),
                    "evidence": category.get("evidence", ""),
                    "primary_business": primary_business,
                    "key_technologies": ", ".join(key_technologies) if isinstance(key_technologies, list) else key_technologies,
                    "key_capabilities": ", ".join(key_capabilities) if isinstance(key_capabilities, list) else key_capabilities,
                    "target_industries": ", ".join(target_industries) if isinstance(target_industries, list) else target_industries,
                    "notable_products": ", ".join(notable_products) if isinstance(notable_products, list) else notable_products,
                    "summary": summary,
                    "github_repos_count": github_repos_count,
                    "top_github_repo": top_github_repo,
                    "top_github_stars": top_github_stars,
                    "analysis_model": analysis_model,
                    "bootstrap_model": bootstrap_model,
                    "data_sources": ", ".join(data_sources),
                    "timestamp": timestamp
                }
                
                all_analysis_results.append(result_entry)
        else:
            print(f"FAILED TO GET ANALYSIS FOR: {partner_name}")
    
    # Save consolidated CSV file if CSV output format is selected
    if (args.output_format == "csv" or args.output_format == "both") and all_analysis_results:
        # Generate timestamp for consolidated file
        consolidated_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_csv_file = os.path.join(args.output_dir, f"partner_analysis_{consolidated_timestamp}.csv")
        
        with open(consolidated_csv_file, "w", newline="") as f:
            # Get all possible field names from all results
            all_fieldnames = set()
            for result in all_analysis_results:
                all_fieldnames.update(result.keys())
            
            # Ensure core fields come first in a specific order
            core_fields = [
                "partner_name", "category", "confidence", "evidence", 
                "primary_business", "summary", "key_technologies", "key_capabilities", 
                "target_industries", "notable_products", 
                "github_repos_count", "top_github_repo", "top_github_stars",
                "analysis_model", "bootstrap_model", "data_sources", "timestamp"
            ]
            
            # Create final fieldnames list with core fields first, then any additional fields
            fieldnames = core_fields + sorted(list(all_fieldnames - set(core_fields)))
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write all results
            for result in all_analysis_results:
                writer.writerow(result)
        
        print(f"SAVED CONSOLIDATED CSV RESULTS TO: {consolidated_csv_file}")
    
    print("MAIN FUNCTION COMPLETED")

if __name__ == "__main__":
    print("SCRIPT ENTRY POINT REACHED")
    
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
    finally:
        print("Cleaning up resources")
