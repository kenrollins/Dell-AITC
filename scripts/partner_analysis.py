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
    --consolidate              Generate a consolidated CSV file from all JSON files in the output directory
"""

import os
import re
import csv
import json
import logging
import asyncio
import argparse
import datetime
import traceback
import urllib.parse
import glob
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

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
    parser = argparse.ArgumentParser(
        description="Process partners and generate analysis results."
    )

    # Input/output options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of partners to process",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="data/input/partners.csv",
        help="Path to the CSV file containing partner data",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default="data/input/AI-Technology-Categories-v1.4.csv",
        help="Path to the CSV file containing AI technology categories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/partner_analysis",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "both"],
        help="Output format (json, csv, or both)",
    )

    # Model options
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="URL for Ollama API",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_ANALYSIS_MODEL,
        help="Model to use for Ollama analysis",
    )
    parser.add_argument(
        "--bootstrap-model",
        type=str,
        default=DEFAULT_BOOTSTRAP_MODEL,
        help="Model to use for bootstrap information",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="Model to use for OpenAI",
    )
    parser.add_argument(
        "--openai-primary", action="store_true", help="Use OpenAI as primary model"
    )
    parser.add_argument(
        "--openai-fallback", action="store_true", help="Use OpenAI as fallback model"
    )

    # Feature flags
    parser.add_argument(
        "--skip-website", action="store_true", help="Skip fetching website content"
    )
    parser.add_argument(
        "--skip-search", action="store_true", help="Skip Brave search enhancement"
    )
    parser.add_argument(
        "--skip-github", action="store_true", help="Skip GitHub search enhancement"
    )

    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Add consolidate argument
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Generate a consolidated CSV file from all JSON files in the output directory.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = (
        logging.DEBUG
        if args.debug
        else (logging.INFO if args.verbose else logging.WARNING)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("partner_analysis.log"), logging.StreamHandler()],
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def _load_ai_categories(categories_file):
    """Load AI technology categories from a CSV file."""
    print(f"Loading AI technology categories from: {categories_file}")

    categories = []

    try:
        with open(
            categories_file, "r", encoding="utf-8-sig"
        ) as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names to ensure consistency
                cleaned_row = {}
                for key, value in row.items():
                    # Remove BOM character if present
                    clean_key = key.replace("\ufeff", "")
                    cleaned_row[clean_key] = value

                # Extract category information
                category = {
                    "name": cleaned_row.get("ai_category", "").strip(),
                    "description": cleaned_row.get("definition", "").strip(),
                    "keywords": cleaned_row.get("keywords", "").strip(),
                    "capabilities": cleaned_row.get("capabilities", "").strip(),
                    "business_language": cleaned_row.get(
                        "business_language", ""
                    ).strip(),
                    "maturity_level": cleaned_row.get("maturity_level", "").strip(),
                    "zone": cleaned_row.get("zone", "").strip(),
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
            description = await page.evaluate(
                """
                () => {
                    const metaDescription = document.querySelector('meta[name="description"]');
                    return metaDescription ? metaDescription.getAttribute('content') : '';
                }
            """
            )

            # Get the page content
            content = await page.evaluate(
                """
                () => {
                    // Remove scripts, styles, and other non-content elements
                    const scripts = document.querySelectorAll('script, style, noscript, iframe, img, svg, canvas, video, audio');
                    scripts.forEach(s => s.remove());
                    
                    // Get the main content
                    const body = document.body;
                    return body.innerText;
                }
            """
            )

            # Clean up the content
            clean_content = re.sub(r"\s+", " ", content).strip()

            # Close the browser
            await browser.close()

            # Return the website content
            return {
                "title": title,
                "description": description,
                "content": clean_content,
                "url": url,
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
        "X-Subscription-Token": brave_api_key,
    }

    # Set up the parameters
    params = {"q": query, "count": 10, "search_lang": "en"}

    # Initialize retry counter
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    api_url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        # Parse the response
                        data = await response.json()

                        # Extract the search results
                        results = []
                        for web in data.get("web", {}).get("results", []):
                            results.append(
                                {
                                    "title": web.get("title", ""),
                                    "description": web.get("description", ""),
                                    "url": web.get("url", ""),
                                    "source": "brave_search",
                                }
                            )

                        print(f"Found {len(results)} search results for {query}")
                        return results
                    elif response.status == 429:
                        # Rate limiting, wait and retry
                        retry_count += 1
                        wait_time = 2**retry_count  # Exponential backoff
                        print(
                            f"Brave Search API rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Other error
                        print(
                            f"Brave Search API returned status code {response.status}"
                        )
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
                                all_results.append(
                                    {
                                        "name": repo.get("name"),
                                        "full_name": repo.get("full_name"),
                                        "description": repo.get("description"),
                                        "url": repo.get("html_url"),
                                        "stars": repo.get("stargazers_count"),
                                        "forks": repo.get("forks_count"),
                                        "language": repo.get("language"),
                                        "topics": repo.get("topics", []),
                                        "source": "github",
                                    }
                                )

                            # Break out of retry loop on success
                            break
                        elif response.status == 403:
                            # Rate limiting, wait and retry
                            retry_count += 1
                            wait_time = 2**retry_count  # Exponential backoff
                            logging.warning(
                                f"GitHub API rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logging.warning(
                                f"GitHub API returned status code {response.status}"
                            )
                            break
            except Exception as e:
                logging.error(f"Error searching GitHub: {str(e)}")
                break

    print(f"Found {len(all_results)} GitHub results for {vendor_name}")
    return all_results


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self, model_name=DEFAULT_BOOTSTRAP_MODEL, host="http://localhost:11434"
    ):
        """Initialize the Ollama client."""
        self.model_name = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        print(f"Initialized Ollama client with model: {model_name}")

    async def generate(self, prompt):
        """Generate a response from Ollama."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {"model": self.model_name, "prompt": prompt, "stream": False}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(
                            f"Ollama API returned status code {response.status}: {error_text}"
                        )
                        raise Exception(
                            f"Ollama API returned status code {response.status}"
                        )

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
                    {
                        "role": "system",
                        "content": "You are an expert technology analyst specializing in AI capabilities assessment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.2,
            )

            # Log the response
            logging.info(
                f"Received response from OpenAI (length: {len(response.choices[0].message.content)} characters)"
            )
            logging.debug(
                f"Response preview: {response.choices[0].message.content[:100]}..."
            )

            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {str(e)}")
            print(f"ERROR GENERATING RESPONSE FROM OPENAI: {str(e)}")
            return None


def parse_llm_json_response(response_text):
    """
    Parse JSON response from LLM, handling various edge cases and formats.
    Returns a cleaned dictionary or None if parsing fails.
    """
    # Function is complex but necessary for robust parsing
    # noqa: C901

    if not response_text:
        return None

    # Try to find JSON content within the response
    json_pattern = r"```(?:json)?(.*?)```"
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)

    if json_matches:
        # Use the first JSON block found
        json_content = json_matches[0].strip()
    else:
        # If no JSON blocks with markers, try to extract JSON directly
        json_content = response_text.strip()

    # Try to find the start and end of JSON object
    start_idx = json_content.find("{")
    end_idx = json_content.rfind("}")

    if start_idx != -1 and end_idx != -1:
        json_content = json_content[start_idx : end_idx + 1]

    # Try to parse the JSON
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        # If parsing fails, try to fix common issues
        try:
            # Replace single quotes with double quotes
            fixed_content = json_content.replace("'", '"')
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            try:
                # Try to fix trailing commas
                fixed_content = re.sub(r",\s*}", "}", json_content)
                fixed_content = re.sub(r",\s*]", "]", fixed_content)
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse LLM response as JSON: {response_text}")
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


def _create_analysis_prompt(
    partner,
    bootstrap_info,
    website_content,
    ai_categories,
    search_results=None,
    github_results=None,
):
    """Create an analysis prompt for a partner."""
    print(f"Creating analysis prompt for {partner}")

    # Format the AI categories
    categories_text = "\n".join(
        [
            f"{i+1}. {cat['name']}: {cat['description']}"
            for i, cat in enumerate(ai_categories)
        ]
    )

    # Format the bootstrap information
    bootstrap_text = (
        json.dumps(bootstrap_info, indent=2)
        if bootstrap_info
        else "No bootstrap information available."
    )

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
        print(
            f"Received bootstrap response for {partner} (length: {len(bootstrap_response)} characters)"
        )

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
        print(
            f"Received bootstrap response for {partner} (length: {len(bootstrap_response)} characters)"
        )

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


async def _get_analysis_ollama(
    partner,
    bootstrap_info,
    website_content,
    ai_categories,
    search_results=None,
    github_results=None,
    model_name=DEFAULT_ANALYSIS_MODEL,
):
    """Get analysis for a partner using Ollama."""
    print(f"Getting analysis for {partner} using Ollama")

    # Create the analysis prompt
    analysis_prompt = _create_analysis_prompt(
        partner,
        bootstrap_info,
        website_content,
        ai_categories,
        search_results,
        github_results,
    )

    # Initialize the Ollama client
    ollama_client = OllamaClient(model_name=model_name)

    try:
        # Generate the analysis response
        analysis_response = await ollama_client.generate(analysis_prompt)
        print(
            f"Received analysis response for {partner} (length: {len(analysis_response)} characters)"
        )

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


async def _get_analysis_openai(
    partner,
    bootstrap_info,
    website_content,
    ai_categories,
    search_results=None,
    github_results=None,
    model_name=DEFAULT_OPENAI_MODEL,
):
    """Get analysis for a partner using OpenAI."""
    print(f"Getting analysis for {partner} using OpenAI")

    # Create the analysis prompt
    analysis_prompt = _create_analysis_prompt(
        partner,
        bootstrap_info,
        website_content,
        ai_categories,
        search_results,
        github_results,
    )

    # Initialize the OpenAI client
    openai_client = OpenAIClient(model_name=model_name)

    try:
        # Generate the analysis response
        analysis_response = await openai_client.generate(analysis_prompt)
        print(
            f"Received analysis response for {partner} (length: {len(analysis_response)} characters)"
        )

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


async def _get_analysis(
    partner,
    bootstrap_info,
    website_content,
    ai_categories,
    search_results=None,
    github_results=None,
    use_openai=False,
):
    """Get analysis for a partner."""
    if use_openai:
        return await _get_analysis_openai(
            partner,
            bootstrap_info,
            website_content,
            ai_categories,
            search_results,
            github_results,
        )
    else:
        return await _get_analysis_ollama(
            partner,
            bootstrap_info,
            website_content,
            ai_categories,
            search_results,
            github_results,
        )


def consolidate_json_to_csv(output_dir):
    """
    Generate a consolidated CSV file from all JSON files in the output directory.
    
    Args:
        output_dir (str): Path to the directory containing JSON files
        
    Returns:
        str: Path to the consolidated CSV file
    """
    logging.info("Consolidating JSON files to CSV...")
    
    # Find all JSON files in the output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    if not json_files:
        logging.warning("No JSON files found in the output directory.")
        return None
    
    # Define the core fields for the CSV
    core_fields = [
        "partner_name", "primary_business", "summary", 
        "key_technologies", "key_capabilities", "target_industries", 
        "notable_products", "matched_categories", "analysis_model",
        "bootstrap_model", "data_sources", "github_repos_count",
        "top_github_repo", "top_github_stars"
    ]
    
    # List to store all results
    all_results = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract partner data
            result_entry = {
                "partner_name": data.get("partner_name", ""),
                "primary_business": data.get("primary_business", ""),
                "summary": data.get("summary", ""),
                "key_technologies": ", ".join(data.get("key_technologies", [])),
                "key_capabilities": ", ".join(data.get("key_capabilities", [])),
                "target_industries": ", ".join(data.get("target_industries", [])),
                "notable_products": ", ".join(data.get("notable_products", [])),
                "matched_categories": data.get("matched_categories", []),
            }
            
            # Extract metadata
            metadata = data.get("analysis_metadata", {})
            result_entry.update({
                "analysis_model": metadata.get("analysis_model", ""),
                "bootstrap_model": metadata.get("bootstrap_model", ""),
                "data_sources": ", ".join(metadata.get("data_sources", [])),
                "github_repos_count": metadata.get("github_repos_count", 0),
                "top_github_repo": metadata.get("top_github_repo", ""),
                "top_github_stars": metadata.get("top_github_stars", 0)
            })
            
            # Process matched categories
            matched_categories = result_entry.get("matched_categories", [])
            for category in matched_categories:
                category_name = category.get("category", "")
                confidence = category.get("confidence", 0)
                result_entry[f"{category_name}_confidence"] = confidence
                
                # Add evidence as a separate column
                evidence = category.get("evidence", "")
                if isinstance(evidence, list):
                    evidence = ", ".join(evidence)
                result_entry[f"{category_name}_evidence"] = evidence
            
            all_results.append(result_entry)
            logging.info(f"Processed {os.path.basename(json_file)}")
            
        except Exception as e:
            logging.error(f"Error processing {json_file}: {str(e)}")
    
    if not all_results:
        logging.warning("No valid results found in JSON files.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Generate timestamp for the consolidated file
    consolidated_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    consolidated_csv_file = os.path.join(output_dir, f"partner_analysis_{consolidated_timestamp}.csv")
    
    # Save to CSV
    df.to_csv(consolidated_csv_file, index=False)
    logging.info(f"Consolidated CSV saved to {consolidated_csv_file}")
    
    return consolidated_csv_file


async def main():
    """
    Main function to process partners and generate AI technology analysis.
    """
    # Function is complex but handles the main workflow
    # noqa: C901

    args = _parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load AI categories
    ai_categories = _load_ai_categories(args.categories_file)

    # Load partners from CSV
    partners = _load_partners_from_csv(args.csv_file, args.limit)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create results file
    results_file = os.path.join(args.output_dir, "results.csv")

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Partner",
                "Website",
                "Description",
                "AI Categories",
                "Confidence",
                "Reasoning",
            ]
        )

    # Process each partner
    for partner in partners:
        try:
            logging.info(f"Processing partner: {partner['partner_name']}")

            # Get bootstrap information
            bootstrap_info = await _get_bootstrap_info(partner, args.openai_primary)

            # Get website content
            website_content = ""
            if partner.get("website_url"):
                try:
                    website_content = await _get_website_content(partner["website_url"])
                except Exception as e:
                    logging.error(f"Error fetching website content: {e}")

            # Get search results
            search_results = None
            if not args.skip_search:
                try:
                    search_results = await _get_brave_search_results(
                        partner["partner_name"]
                    )
                except Exception as e:
                    logging.error(f"Error fetching search results: {e}")

            # Get GitHub results
            github_results = None
            if not args.skip_github:
                try:
                    github_results = await _search_github(partner["partner_name"])
                except Exception as e:
                    logging.error(f"Error fetching GitHub results: {e}")

            # Get analysis
            analysis = await _get_analysis(
                partner,
                bootstrap_info,
                website_content,
                ai_categories,
                search_results,
                github_results,
                args.openai_primary,
            )

            # Write results to CSV
            if analysis:
                with open(results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            partner["partner_name"],
                            partner.get("website_url", ""),
                            analysis.get("description", ""),
                            ", ".join(analysis.get("matched_categories", [])),
                            analysis.get("confidence", ""),
                            analysis.get("reasoning", ""),
                        ]
                    )

                # Write detailed analysis to JSON file
                partner_file = os.path.join(
                    args.output_dir, f"{partner['partner_name'].replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(partner_file, "w") as f:
                    # Create a comprehensive JSON structure that includes all required fields
                    partner_data = {
                        "partner_name": partner['partner_name'],
                        "matched_categories": analysis.get("matched_categories", []),
                        "primary_business": bootstrap_info.get("primary_business", ""),
                        "summary": bootstrap_info.get("summary", ""),
                        "key_technologies": bootstrap_info.get("key_technologies", []),
                        "key_capabilities": bootstrap_info.get("key_capabilities", []),
                        "target_industries": bootstrap_info.get("target_industries", []),
                        "notable_products": bootstrap_info.get("notable_products", []),
                        "analysis_metadata": {
                            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "analysis_model": args.ollama_model if not args.openai_primary else args.openai_model,
                            "bootstrap_model": args.bootstrap_model,
                            "data_sources": ["website" if website_content else None, 
                                            "search" if search_results else None, 
                                            "github" if github_results else None],
                            "search_results_count": len(search_results) if search_results else 0,
                            "github_repos_count": len(github_results) if github_results else 0,
                            "top_github_repo": github_results[0].get("full_name", "") if github_results and len(github_results) > 0 else "",
                            "top_github_stars": github_results[0].get("stars", 0) if github_results and len(github_results) > 0 else 0
                        }
                    }
                    
                    # Remove None values from data_sources
                    partner_data["analysis_metadata"]["data_sources"] = [source for source in partner_data["analysis_metadata"]["data_sources"] if source]
                    
                    json.dump(
                        partner_data,
                        f,
                        indent=2,
                    )

            logging.info(f"Completed analysis for {partner['partner_name']}")

            # Sleep to avoid rate limiting
            if not args.openai_primary:
                await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"Error processing partner {partner['partner_name']}: {e}")
            traceback.print_exc()

    logging.info("Analysis complete")

    # Generate consolidated CSV from JSON files
    if args.consolidate:
        consolidate_json_to_csv(args.output_dir)


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
