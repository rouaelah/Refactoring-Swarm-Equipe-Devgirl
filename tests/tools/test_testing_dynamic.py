"""Test dynamic test execution"""
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
    execute_dynamic_tests,
    generate_and_execute_tests,
    benchmark_test_performance,
    _sanitize_dynamic_code,
    _generate_simple_tests_from_spec,
    TestingError,
    SecurityError,
)
from src.tools.security import get_sandbox_root

# Setup
SANDBOX = get_sandbox_root()


def test_sanitize_dynamic_code():
    """Test code sanitization"""
    # Safe code should pass through
    safe_code = '''
def add(a, b):
    return a + b
'''
    sanitized = _sanitize_dynamic_code(safe_code)
    assert "def add" in sanitized
    assert sanitized == safe_code
    
    # Dangerous code should be sanitized
    dangerous_code = '''
import os
os.system("rm -rf /")
eval("print('dangerous')")
'''
    sanitized = _sanitize_dynamic_code(dangerous_code)

    assert "import os" not in sanitized
    assert "os.system" not in sanitized
    assert "eval" not in sanitized


def test_generate_simple_tests_from_spec():
    """Test basic test generation"""
    code = '''
def calculate_sum(a, b):
    return a + b
    
def is_positive(n):
    return n > 0
'''
    
    spec = {
        "description": "Math utility functions",
        "functions": ["calculate_sum", "is_positive"]
    }
    
    test_code = _generate_simple_tests_from_spec(code, spec)
    
    assert "Dynamically generated tests" in test_code
    assert "test_calculate_sum_basic" in test_code
    assert "test_is_positive_basic" in test_code
    assert "import pytest" in test_code


@patch('src.tools.testing._run_pytest_command')
def test_execute_dynamic_tests_success(mock_pytest):
    """Test successful dynamic test execution"""
    code = '''
def add(a, b):
    return a + b
'''
    
    test_code = '''
def test_add():
    from code_under_test import add
    assert add(2, 3) == 5
'''
    
    # Mock successful test execution
    mock_pytest.return_value = {
        "success": True,
        "total_tests": 1,
        "passed": 1,
        "failed": 0,
        "summary": "1 test passed"
    }
    
    results = execute_dynamic_tests(code, test_code)
    
    assert results["success"] is True
    assert results["total_tests"] == 1
    assert mock_pytest.called


def test_execute_dynamic_tests_dangerous_code():
    """Test that dangerous code is rejected"""
    dangerous_test = '''
import os
os.system("rm -rf /")
'''
    
    code = "def dummy(): pass"
    
    with pytest.raises(SecurityError):
        execute_dynamic_tests(code, dangerous_test)


@patch('src.tools.testing.execute_dynamic_tests')
def test_generate_and_execute_tests(mock_execute):
    """Test test generation and execution"""
    code = "def func(): return 42"
    spec = {"description": "Simple function"}
    
    # Mock execution
    mock_execute.return_value = {
        "success": True,
        "total_tests": 2,
        "passed": 2
    }
    
    results = generate_and_execute_tests(code, spec)
    
    assert results["success"] is True
    mock_execute.assert_called_once()


def test_execute_dynamic_tests_timeout():
    """Test timeout handling in dynamic tests"""
    code = '''
def slow_function():
    import time
    time.sleep(20)  # Very slow
    return True
'''
    
    test_code = '''
def test_slow():
    from code_under_test import slow_function
    assert slow_function() is True
'''
    
    # Mock timeout
    with patch('src.tools.testing._run_pytest_command') as mock_pytest:
        mock_pytest.return_value = {
            "success": False,
            "timeout_occurred": True,
            "error": "Test execution timeout",
            "summary": "TIMEOUT"
        }
        
        results = execute_dynamic_tests(code, test_code, timeout=1)
        
        assert results["success"] is False
        assert results["timeout_occurred"] is True


def test_benchmark_test_performance():
    """Test performance benchmarking"""
    code = "def fast(): return 1"
    
    test_cases = [
        "def test1(): assert 1 == 1",
        "def test2(): assert True is True",
    ]
    
    with patch('src.tools.testing.execute_dynamic_tests') as mock_execute:
        # Mock successful test executions
        mock_execute.return_value = {
            "success": True,
            "total_tests": 1,
            "passed": 1
        }
        
        benchmark = benchmark_test_performance(code, test_cases, iterations=2)
        
        assert benchmark["total_tests"] == 2
        assert benchmark["iterations"] == 2
        assert len(benchmark["test_results"]) == 2
        assert "performance" in benchmark
        assert "average_time" in benchmark["performance"]


def test_execute_dynamic_tests_syntax_error():
    """Test handling of syntax errors in test code"""
    code = "def good(): return 1"
    
    test_with_syntax_error = '''
def test_broken(
    # Missing closing parenthesis
'''
    
    with pytest.raises(Exception):  # Should raise TestingError or SecurityError
        execute_dynamic_tests(code, test_with_syntax_error)


def test_safe_environment_creation():
    """Test that safe environments are created within sandbox"""
    from src.tools.testing import _create_safe_test_environment
    
    with _create_safe_test_environment() as temp_dir:
        temp_path = Path(temp_dir)
        sandbox_root = get_sandbox_root()
        
        # Ensure temp directory is within sandbox
        assert str(temp_path).startswith(str(sandbox_root))
        
        # Ensure it has safe prefix
        assert temp_path.name.startswith("safe_test_")


@patch('src.tools.testing.log_experiment')
def test_dynamic_tests_logging(mock_logger):
    """Test that dynamic test execution is logged"""
    code = "def test(): pass"
    test_code = "def test_dyn(): assert True"
    
    with patch('src.tools.testing._run_pytest_command') as mock_pytest:
        mock_pytest.return_value = {
            "success": True,
            "total_tests": 1,
            "passed": 1
        }
        
        execute_dynamic_tests(code, test_code)
        
        # Verify logging was called
        assert mock_logger.called
        call_args = mock_logger.call_args
        print(call_args)
        assert call_args[1]['details']['operation'] == "execute_dynamic_tests"
        assert call_args[1]['status'] == "SUCCESS"


def test_real_dynamic_execution():
    """Integration test with real dynamic execution"""
    # Skip if pytest not available
    try:
        import pytest
    except ImportError:
        pytest.skip("pytest not installed")
    
    # Simple valid test
    code = '''
def multiply(a, b):
    return a * b
'''
    
    test_code = '''
def test_multiply():
    from code_under_test import multiply
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6
'''
    
    results = execute_dynamic_tests(code, test_code, timeout=5)
    
    assert "success" in results
    assert results.get("total_tests", 0) >= 1
    
    # Test with failing test
    failing_test = '''
def test_failing():
    from code_under_test import multiply
    assert multiply(2, 2) == 5  # Wrong!
'''
    
    results = execute_dynamic_tests(code, failing_test, timeout=5)
    assert results.get("return_code", 0) ==1  # Should fail


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])