#!/usr/bin/env python3
"""
Test runner script for infinite tensor tests.

This script provides an easy way to run tests with different configurations.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0


def main():
    """Main test runner function."""
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    
    # Python executable to use
    python_cmd = "python3.11"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python3.11 run_tests.py [option]")
            print("\nOptions:")
            print("  --help, -h     Show this help message")
            print("  --verbose, -v  Run tests with verbose output")
            print("  --coverage     Run tests with coverage report")
            print("  --fast         Run tests without coverage")
            print("  --class <name> Run specific test class")
            print("  --method <name> Run specific test method")
            print("\nExamples:")
            print("  python3.11 run_tests.py")
            print("  python3.11 run_tests.py --verbose")
            print("  python3.11 run_tests.py --coverage")
            print("  python3.11 run_tests.py --class TestInfiniteTensorBasics")
            return
        
        elif sys.argv[1] == "--verbose" or sys.argv[1] == "-v":
            cmd = [python_cmd, "-m", "pytest", "tests/", "-v"]
            
        elif sys.argv[1] == "--coverage":
            cmd = [python_cmd, "-m", "pytest", "tests/", "--cov=infinite_tensors", "--cov-report=term-missing", "--cov-report=html"]
            
        elif sys.argv[1] == "--fast":
            cmd = [python_cmd, "-m", "pytest", "tests/", "-v", "--tb=short"]
            
        elif sys.argv[1] == "--class" and len(sys.argv) > 2:
            class_name = sys.argv[2]
            cmd = [python_cmd, "-m", "pytest", f"tests/test_infinite_tensor.py::{class_name}", "-v"]
            
        elif sys.argv[1] == "--method" and len(sys.argv) > 2:
            method_name = sys.argv[2]
            cmd = [python_cmd, "-m", "pytest", "-k", method_name, "-v"]
            
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for available options")
            return
    else:
        # Default: run all tests with basic output
        cmd = [python_cmd, "-m", "pytest", "tests/", "-v", "--tb=short"]
    
    # Run the tests
    success = run_command(cmd)
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
