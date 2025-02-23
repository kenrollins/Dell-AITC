"""
Test OpenAI Connection

This script tests the OpenAI API connection using credentials from .env file.

Usage:
    python test_openai_connection.py

Additional Info:
    Requires .env file with OPENAI_API_KEY and OPENAI_MODEL configured
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
from pathlib import Path

def setup_environment():
    """Load environment variables from .env file."""
    # Get the project root directory (2 levels up from scripts/)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    required_vars = ['OPENAI_API_KEY', 'OPENAI_MODEL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

def test_openai_connection():
    """Test OpenAI API connection with a simple request."""
    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=float(os.getenv('OPENAI_TIMEOUT', '60.0'))
        )
        
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please respond with 'Connection successful!'"}
            ],
            max_tokens=50
        )
        
        print("API Connection Test Results:")
        print("-" * 50)
        print(f"Model: {os.getenv('OPENAI_MODEL')}")
        print(f"Response: {response.choices[0].message.content}")
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"Error connecting to OpenAI API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting OpenAI connection test...")
    setup_environment()
    success = test_openai_connection()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1) 