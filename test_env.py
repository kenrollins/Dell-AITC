import sys
import os

print("Environment Information:")
print("-" * 50)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"\nPYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"PYTHONHOME: {os.environ.get('PYTHONHOME', 'Not set')}")
print("\nSystem Path:")
for path in sys.path:
    print(f"  - {path}") 