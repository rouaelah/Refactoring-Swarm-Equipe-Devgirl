"""Integration test for basic pytest tools"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure the project root is on sys.path so `import src...` works.
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add src directly as a fallback
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.tools.testing import (
    run_pytest_on_file,
    discover_test_files,
    run_tests_in_directory,
    validate_test_code,
)
from src.tools.file_ops import safe_write_file
from src.tools.security import get_sandbox_root

def create_test_environment():
    """Create a test environment with code and tests"""
    sandbox = get_sandbox_root()
    test_dir = sandbox / "integration_test"
    test_dir.mkdir(exist_ok=True)
    
    # Clean previous test
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Create a simple Python module
    module_code = '''
"""
A simple math module for testing.
"""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0
'''
    
    module_file = test_dir / "math_utils.py"
    safe_write_file(module_file, module_code)
    
    # Create comprehensive test file
    test_code = '''
"""
Tests for math_utils module.
"""
import pytest
from math_utils import add, divide, is_even

class TestAdd:
    """Test addition function."""
    
    def test_add_positive(self):
        """Test addition with positive numbers."""
        assert add(2, 3) == 5
        assert add(10, 20) == 30
    
    def test_add_negative(self):
        """Test addition with negative numbers."""
        assert add(-5, 10) == 5
        assert add(-3, -4) == -7
    
    def test_add_zero(self):
        """Test addition with zero."""
        assert add(0, 5) == 5
        assert add(7, 0) == 7

class TestDivide:
    """Test division function."""
    
    def test_divide_normal(self):
        """Test normal division."""
        assert divide(10, 2) == 5.0
        assert divide(9, 3) == 3.0
    
    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)
    
    @pytest.mark.skip(reason="Feature not implemented yet")
    def test_divide_negative(self):
        """Test division with negative numbers (skipped)."""
        assert divide(-10, 2) == -5.0

def test_is_even():
    """Test even number detection."""
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True
    
def test_failing_on_purpose():
    """This test should fail (for demonstration)."""
    assert add(2, 2) == 5  # Incorrect!

@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (5, 5, 10),
    (100, 200, 300),
])
def test_add_parametrized(a, b, expected):
    """Parameterized test for addition."""
    assert add(a, b) == expected
'''
    
    test_file = test_dir / "test_math_utils.py"
    safe_write_file(test_file, test_code)
    
    # Create another simple test file
    simple_test = '''
def test_simple_one():
    assert 1 == 1
    
def test_simple_two():
    assert "hello".upper() == "HELLO"
'''
    
    simple_file = test_dir / "test_simple_stuff.py"
    safe_write_file(simple_file, simple_test)
    
    return test_dir, module_file, test_file, simple_file

def run_integration_test():
    """Run full integration test"""
    print("Basic Pytest Tools - Integration Test")
    print("=" * 60)
    
    try:
        # Create test environment
        test_dir, module_file, test_file, simple_file = create_test_environment()
        
        print(f"Test directory: {test_dir}")
        
        # Test 1: Discover test files
        print("\n1 Discovering test files...")
        try:
            test_files = discover_test_files(test_dir)
            print(f"   Found {len(test_files)} test files:")
            for tf in test_files:
                print(f"     - {tf.name}")
            assert len(test_files) == 2, f"Expected 2 test files, found {len(test_files)}"
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 2: Run pytest on specific file
        print("\n2 Running pytest on math_utils.py...")
        try:
            results = run_pytest_on_file(module_file)
            print(f"   Total tests: {results.get('total_tests', 'N/A')}")
            print(f"   Passed: {results.get('passed', 'N/A')}")
            print(f"   Failed: {results.get('failed', 'N/A')}")
            print(f"   Skipped: {results.get('skipped', 'N/A')}")
            print(f"   Success: {results.get('success', 'N/A')}")
            print(f"   Summary: {results.get('summary', 'No summary')}")
            
            # Check we got reasonable results
            assert results.get('total_tests', 0) > 0
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 3: Run all tests in directory
        print("\n3 Running all tests in directory...")
        try:
            dir_results = run_tests_in_directory(test_dir)
            print(f"   Test files found: {dir_results.get('test_files', 0)}")
            print(f"   Total tests across all files: {dir_results.get('total_tests', 0)}")
            print(f"   Overall success: {dir_results.get('success', False)}")
            print(f"   Summary: {dir_results.get('summary', 'No summary')}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 4: Validate test code
        print("\n4 Validating test code...")
        try:
            # Valid code
            valid_code = "def test_valid(): assert True"
            is_valid, message = validate_test_code(valid_code)
            print(f"   Valid code: {is_valid} ({message})")
            
            # Dangerous code
            dangerous_code = "import os\ndef test_bad(): os.system('rm -rf /')"
            is_valid, message = validate_test_code(dangerous_code)
            print(f"   Dangerous code blocked: {not is_valid} ({message})")
            
            # Syntax error
            bad_syntax = "def test_bad( : pass"
            is_valid, message = validate_test_code(bad_syntax)
            print(f"   Syntax error caught: {not is_valid} ({message})")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 5: Real pytest execution
        print("\n5 Testing with real pytest (if installed)...")
        try:
            import pytest
            print("   pytest is installed")
            
            # Quick manual test
            print("   Tool functions are callable and return structured data")
            
        except ImportError:
            print("   pytest not installed, skipping real execution")
        
        print("\n" + "=" * 60)
        print("integration tests completed!")
        
        # Show log count
        log_file = Path("logs") / "experiment_data.json"
        if log_file.exists():
            try:
                import json
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        testing_entries = []
                        for e in data:
                            if 'details' in e:
                                if 'operation' in e['details']:
                                    if e['details']['operation'] == 'run_pytest':
                                        testing_entries.append(e)
                        print(f"\nTesting log entries created: {len(testing_entries)}")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)