"""Test basic pytest integration"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call


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
    TestingError,
    _parse_pytest_output,
)
from src.tools.security import get_sandbox_root

# Setup
SANDBOX = get_sandbox_root()


@pytest.fixture
def test_environment():
    """Create test files with tests"""
    test_dir = SANDBOX / "test_testing"
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple Python module
    module_code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b
'''
    
    module_file = test_dir / "calculator.py"
    module_file.write_text(module_code)
    
    # Create test file
    test_code = '''
import pytest
from calculator import add, subtract, multiply

def test_add():
    """Test addition."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    """Test subtraction."""
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5

def test_multiply():
    """Test multiplication."""
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    
def test_failing():
    """This test should fail."""
    assert add(2, 2) == 5  # Wrong!

@pytest.mark.skip
def test_skipped():
    """This test is skipped."""
    assert True
'''
    
    test_file = test_dir / "test_calculator.py"
    test_file.write_text(test_code)
    
    # Create another test file
    test2_code = '''
def test_simple():
    """Simple test."""
    assert 1 + 1 == 2
'''
    
    test2_file = test_dir / "test_simple.py"
    test2_file.write_text(test2_code)
    
    yield test_dir, module_file, test_file, test2_file


def test_parse_pytest_output():
    """Test parsing pytest output"""
    # Sample pytest output
    output = '''
============================= test session starts =============================
platform linux -- Python 3.9.0, pytest-7.0.0, pluggy-1.0.0
rootdir: /sandbox
collected 5 items

test_calculator.py ....F                                                    [100%]

================================== FAILURES ===================================
_______________________________ test_failing __________________________________
    def test_failing():
        """This test should fail."""
>       assert add(2, 2) == 5  # Wrong!
E       assert 4 == 5

test_calculator.py:20: AssertionError
=========================== short test summary info ===========================
FAILED test_calculator.py::test_failing - assert 4 == 5
======================= 1 failed, 4 passed in 0.12s ===========================
'''
    
    result = _parse_pytest_output(output)
    
    assert result["total_tests"] == 5
    assert result["passed"] == 4
    assert result["failed"] == 1
    assert result["duration"] == 0.12
    #print(result['summary'])
    assert "4 passed, 1 failed" in result["summary"]


def test_parse_json_output():
    """Test parsing JSON pytest output"""
    json_output = '''{
  "report": {
    "environment": {},
    "tests": [
      {
        "nodeid": "test_calculator.py::test_add",
        "outcome": "passed",
        "duration": 0.001
      }
    ],
    "summary": {
      "total": 1,
      "passed": 1,
      "failed": 0,
      "skipped": 0,
      "error": 0,
      "duration": 0.001
    }
  }
}'''
    
    result = _parse_pytest_output(json_output)
    assert result["total_tests"] == 1
    assert result["passed"] == 1


@patch('src.tools.testing._run_pytest_command')
def test_run_pytest_on_file(mock_pytest, test_environment):
    """Test running pytest on a file"""
    test_dir, module_file, test_file, test2_file = test_environment
    
    # Mock pytest response
    mock_pytest.return_value = {
        "success": True,
        "total_tests": 5,
        "passed": 4,
        "failed": 1,
        "skipped": 1,
        "errors": 0,
        "duration": 0.12,
        "summary": "5 tests: 4 passed, 1 failed"
    }
    
    results = run_pytest_on_file(module_file)
    
    assert results["success"] is True
    assert results["total_tests"] == 5
    assert results["passed"] == 4
    assert results["failed"] == 1
    
    # Verify mock was called with correct arguments
    mock_pytest.assert_called_once()


def test_run_pytest_on_file_security():
    """Test security in pytest execution"""
    with pytest.raises(Exception):  # Should be SecurityError or TestingError
        run_pytest_on_file("/etc/passwd")


def test_discover_test_files(test_environment):
    """Test test file discovery"""
    test_dir, module_file, test_file, test2_file = test_environment
    
    test_files = discover_test_files(test_dir)
    
    # Should find both test files
    assert len(test_files) == 2
    file_names = [f.name for f in test_files]
    assert "test_calculator.py" in file_names
    assert "test_simple.py" in file_names
    
    # Should not find non-test files
    assert "calculator.py" not in file_names


def test_run_tests_in_directory(test_environment):
    """Test running all tests in a directory"""
    test_dir, module_file, test_file, test2_file = test_environment
    
    with patch('src.tools.testing.run_pytest_on_file') as mock_run:
        # Mock individual test runs
        mock_run.side_effect = [
            {"success": True, "total_tests": 5, "passed": 4, "failed": 1},
            {"success": True, "total_tests": 1, "passed": 1, "failed": 0},
        ]
        
        results = run_tests_in_directory(test_dir)
        
        assert results["test_files"] == 2
        assert results["files_tested"] == 2
        assert results["total_tests"] == 6  # 5 + 1
        assert results["passed"] == 5  # 4 + 1
        assert results["failed"] == 1


def test_validate_test_code():
    """Test test code validation"""
    # Valid code
    valid_code = '''
def test_valid():
    assert 1 + 1 == 2
'''
    is_valid, message = validate_test_code(valid_code)
    assert is_valid is True
    assert "valid" in message.lower()
    
    # Dangerous import
    dangerous_code = '''
import os
def test_dangerous():
    os.system("rm -rf /")
'''
    is_valid, message = validate_test_code(dangerous_code)
    assert is_valid is False
    assert "dangerous" in message.lower()
    
    # Syntax error
    bad_syntax = '''
def test_bad(
    # Missing closing parenthesis
'''
    is_valid, message = validate_test_code(bad_syntax)
    assert is_valid is False
    assert "syntax" in message.lower()


def test_empty_directory():
    """Test running tests in empty directory"""
    empty_dir = SANDBOX / "empty_test_dir"
    empty_dir.mkdir(exist_ok=True)
    
    results = run_tests_in_directory(empty_dir)
    
    assert results["success"] is True
    assert results["total_tests"] == 0
    assert results["test_files"] == 0
    assert "No test files found" in results["summary"]


def test_logging_integration(test_environment):
    """Test that testing operations are logged"""
    test_dir, module_file, test_file, test2_file = test_environment
    
    with patch('src.tools.testing._run_pytest_command') as mock_pytest:
        mock_pytest.return_value = {
            "success": True,
            "total_tests": 5,
            "summary": "Tests passed"
        }
        
        with patch('src.tools.testing.log_experiment') as mock_log:
            run_pytest_on_file(module_file)
            
            # Verify logging was called
            assert mock_log.called
            call_args = mock_log.call_args
            assert call_args[1]['agent_name'] == "Toolsmith"
            assert call_args[1]['status'] == "SUCCESS"


def test_real_pytest_execution():
    """Integration test with real pytest (if available)"""
    try:
        import pytest as real_pytest
    except ImportError:
        pytest.skip("pytest not installed")
    
    # Create a simple test file
    test_dir = SANDBOX / "real_pytest_test"
    test_dir.mkdir(exist_ok=True)
    
    test_code = '''
def test_real():
    """Real test."""
    assert 1 == 1
    
def test_another():
    """Another test."""
    assert 2 + 2 == 4
'''
    
    test_file = test_dir / "test_real.py"
    test_file.write_text(test_code)
    
    # Run pytest
    results = run_pytest_on_file(test_file)
    
    # Verify structure
    assert "success" in results
    assert "total_tests" in results
    assert results["total_tests"] == 2
    assert results["passed"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])