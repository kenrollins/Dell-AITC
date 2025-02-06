import os
from datetime import datetime
import json

def main():
    # Create test data
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "test_message": "Python environment is working correctly!",
        "python_version": os.sys.version
    }
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write test output
    output_file = os.path.join(output_dir, "python_test_output.json")
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test completed successfully!")
    print(f"Output written to: {output_file}")
    print(f"Python version: {os.sys.version}")

if __name__ == "__main__":
    main() 