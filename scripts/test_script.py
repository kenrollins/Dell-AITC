"""
import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime

def log_to_file(message):
    with open('test_output.log', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - {message}\n")

async def test_function():
    log_to_file("Test function started")
    await asyncio.sleep(1)
    log_to_file("Test function completed")

async def main():
    log_to_file("Main function started")
    load_dotenv()
    log_to_file(f"Environment variables loaded - NEO4J_URI: {os.getenv('NEO4J_URI')}")
    await test_function()
    log_to_file("Main function completed")

if __name__ == "__main__":
    log_to_file("Script starting")
    try:
        asyncio.run(main())
    except Exception as e:
        log_to_file(f"Error: {str(e)}")
        import traceback
        log_to_file(traceback.format_exc())
    log_to_file("Script finished")
""" 