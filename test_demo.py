#!/usr/bin/env python3
"""
Test script to validate LoRA demo functionality
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_file_structure():
    """Test that all required files and directories exist."""
    print("🧪 Testing file structure...")
    
    required_files = [
        "README.md",
        "setup.sh", 
        "requirements.txt",
        "src/simple_lora_demo.py",
        "notebooks/01_lora_concepts.ipynb"
    ]
    
    required_dirs = [
        "src",
        "notebooks", 
        "data",
        "models",
        "docs",
        "examples"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    for dir in required_dirs:
        if not Path(dir).exists():
            missing_dirs.append(dir)
    
    if missing_files or missing_dirs:
        print(f"❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required files and directories present")
    return True

def test_python_script():
    """Test that the main Python script runs without errors."""
    print("🧪 Testing Python script execution...")
    
    try:
        result = subprocess.run(
            ["python3", "src/simple_lora_demo.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Python script executed successfully")
            
            # Check if dataset was created
            if Path("data/demo_dataset.json").exists():
                print("✅ Demo dataset created successfully")
            else:
                print("⚠️  Demo dataset not found")
            
            return True
        else:
            print(f"❌ Python script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Python script timed out")
        return False
    except Exception as e:
        print(f"❌ Error running Python script: {e}")
        return False

def test_ollama_availability():
    """Test if Ollama is available and working."""
    print("🧪 Testing Ollama availability...")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ Ollama available: {result.stdout.strip()}")
            
            # Check if model is available
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "llama3.2:1b" in result.stdout:
                print("✅ Llama 3.2 1B model available")
                return True
            else:
                print("⚠️  Llama 3.2 1B model not found")
                return False
                
        else:
            print("❌ Ollama not available")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

def test_dataset_validity():
    """Test if the generated dataset is valid JSON."""
    print("🧪 Testing dataset validity...")
    
    dataset_path = Path("data/demo_dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset file not found")
        return False
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ Dataset valid with {len(data)} examples")
            
            # Check structure of first example
            if all(key in data[0] for key in ['instruction', 'input', 'output']):
                print("✅ Dataset structure correct")
                return True
            else:
                print("❌ Dataset structure invalid")
                return False
        else:
            print("❌ Dataset empty or invalid format")
            return False
            
    except json.JSONDecodeError:
        print("❌ Dataset is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def test_notebook_validity():
    """Test if the Jupyter notebook is valid."""
    print("🧪 Testing notebook validity...")
    
    notebook_path = Path("notebooks/01_lora_concepts.ipynb")
    if not notebook_path.exists():
        print("❌ Notebook file not found")
        return False
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        if 'cells' in notebook and len(notebook['cells']) > 0:
            print(f"✅ Notebook valid with {len(notebook['cells'])} cells")
            return True
        else:
            print("❌ Notebook structure invalid")
            return False
            
    except json.JSONDecodeError:
        print("❌ Notebook is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Running LoRA Demo Validation Tests")
    print("=" * 40)
    
    tests = [
        test_file_structure,
        test_python_script,
        test_dataset_validity,
        test_notebook_validity,
        test_ollama_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Demo is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
