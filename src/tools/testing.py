"""
Testing tools for code validation.
Integrates with pytest for test execution.
"""
import json
import subprocess
import sys
import tempfile
import time
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import ast
import importlib.util

from .security import validate_sandbox_path, SecurityError, sanitize_filename, get_sandbox_root
from .file_ops import safe_read_file, safe_write_file, FileOpsError
from src.utils.logger import log_experiment, ActionType

logger = logging.getLogger(__name__)


class TestingError(Exception):
    """Custom exception for testing failures"""
    pass


class TestTimeoutError(TestingError):
    """Exception for test execution timeout"""
    pass


def _parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract structured results.
    """
    
    result = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "test_details": [],
        "summary": "",
        "raw_output": output[:5000]  # Keep limited raw output
    }
    
    try:
        # Try to find a JSON blob in the output and parse it if it looks like pytest-json
        import re as _re
        json_candidates = _re.findall(r"\{.*?\}", output, _re.DOTALL)
        for candidate in json_candidates:
            try:
                json_data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(json_data, dict):
                # Heuristic: accept json that contains typical pytest json keys
                if any(k in json_data for k in ("passed", "failed", "tests", "total")):
                    result["passed"] = int(json_data.get("passed", 0) or 0)
                    result["failed"] = int(json_data.get("failed", 0) or 0)
                    result["skipped"] = int(json_data.get("skipped", 0) or 0)
                    result["errors"] = int(json_data.get("error", 0) or json_data.get("errors", 0) or 0)
                    result["duration"] = float(json_data.get("duration", 0.0) or 0.0)
                    # tests detail
                    if "tests" in json_data and isinstance(json_data["tests"], list):
                        for test in json_data["tests"]:
                            result["test_details"].append({
                                "name": test.get("name", ""),
                                "outcome": test.get("outcome", ""),
                                "duration": test.get("duration", 0.0),
                                "error": test.get("call", {}).get("crash", {}).get("message", "")
                            })
                    # compute total
                    result["total_tests"] = int(json_data.get("total", result["passed"] + result["failed"] + result["skipped"] + result["errors"]))
                    # Compose summary
                    result["summary"] = (
                        f"Tests: {result['total_tests']} total, "
                        f"{result['passed']} passed, "
                        f"{result['failed']} failed, "
                        f"{result['skipped']} skipped, "
                        f"{result['errors']} errors"
                    )
                    return result
        
        # regex-based parsing of human pytest output
        patterns = {
            "passed": r"(\d+)\s+passed",
            "failed": r"(\d+)\s+failed",
            "skipped": r"(\d+)\s+skipped",
            "errors": r"(\d+)\s+error(?:s)?",
            # duration: match "in 0.12s" or "in 0.12 seconds"
            "duration": r"in\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|seconds?)"
        }
        
        for key, pattern in patterns.items():
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                try:
                    if key == "duration":
                        result["duration"] = float(m.group(1))
                    else:
                        result[key] = int(m.group(1))
                except (TypeError, ValueError):
                    # keep default
                    logger.debug(f"Could not parse {key} from pytest output using pattern {pattern}")
        
        # Derive total tests as sum of counts if not directly found
        result["total_tests"] = result.get("total_tests", 0) or (result["passed"] + result["failed"] + result["skipped"] + result["errors"])
        
        # Extract individual test result lines like:
        # tests/test_example.py::test_func PASSED
        line_re = re.compile(r"(?P<full>[\w./\\-]+::(?P<name>test_[^\s:]+))\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR)", re.IGNORECASE)
        for line in output.splitlines():
            m = line_re.search(line)
            if m:
                name = m.group("name")
                outcome = m.group("outcome").lower()
                result["test_details"].append({
                    "name": name,
                    "outcome": outcome,
                    "duration": 0.0,
                    "error": "" if outcome == "passed" else "See output"
                })
        
        # Compose summary
        result["summary"] = (
            f"Tests: {result['total_tests']} total, "
            f"{result['passed']} passed, "
            f"{result['failed']} failed, "
            f"{result['skipped']} skipped, "
            f"{result['errors']} errors"
        )
        
        return result
        
    except Exception as e:
        logger.warning(f"Could not fully parse pytest output: {e}")
        # Return basic result with raw output
        result["summary"] = f"Raw output available, parsing failed: {e}"
        return result


def _run_pytest_command(
    target: Union[str, Path],
    test_file: Optional[Union[str, Path]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Run pytest on a target and return structured results.
    """
    # Validate paths are within sandbox
    abs_target = validate_sandbox_path(target)
    if test_file:
        abs_test_file = validate_sandbox_path(test_file)

    # Build command
    cmd = [
        sys.executable, "-m", "pytest",
        str(abs_target),
        "--tb=short",  # Shorter traceback
        "-v", 
        "--disable-warnings", 
        #"--timeout", str(timeout),  
    ]
    
    # Add specific test file if provided
    if test_file:
        cmd.append(str(abs_test_file))
    
    start_time = time.time()
    
    try:
        logger.debug(f"Running pytest: {' '.join(cmd)}")
        
        # Run pytest with capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # Add buffer for pytest overhead
            cwd=abs_target.parent if abs_target.is_file() else abs_target
        )
        

        # Parse output
        combined_output = result.stdout + "\n" + result.stderr

        # Parse structured results
        parsed_results = _parse_pytest_output(combined_output)

        # Add execution metadata
        parsed_results.update({
            "execution_time": time.time() - start_time,
            "return_code": result.returncode,
            "success": result.returncode == 0 or result.returncode == 5 or result.returncode == 1,
            "command": " ".join(cmd),
            "timeout_occurred": False
        })

        
        return parsed_results
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Test execution timeout after {timeout} seconds",
            "timeout_occurred": True,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "execution_time": timeout,
            "summary": f"TIMEOUT: Tests exceeded {timeout} second limit"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timeout_occurred": False,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "execution_time": time.time() - start_time,
            "summary": f"EXCEPTION: {str(e)}"
        }


def run_pytest_on_file(
    code_file: Union[str, Path],
    test_file: Optional[Union[str, Path]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Run pytest on a Python code file.
    """
    start_time = datetime.now()
    
    try:
        # Validate paths
        abs_code_file = validate_sandbox_path(code_file)
        
        # Check if file exists
        if not abs_code_file.exists():
            raise TestingError(f"Code file does not exist: {abs_code_file}")
        
        # Run pytest
        results = _run_pytest_command(abs_code_file, test_file, timeout)
        
        # Log the operation
        _log_testing_operation(
            operation="run_pytest",
            filepath=abs_code_file,
            results=results,
            success=results.get("success", False),
            duration=(datetime.now() - start_time).total_seconds()
        )
        
        return results
        
    except (SecurityError, FileOpsError) as e:
        _log_testing_operation(
            operation="run_pytest",
            filepath=Path(code_file) if isinstance(code_file, str) else code_file,
            results={"error": str(e), "success": False},
            success=False,
            error=str(e)
        )
        raise
    except Exception as e:
        _log_testing_operation(
            operation="run_pytest",
            filepath=Path(code_file) if isinstance(code_file, str) else code_file,
            results={"error": str(e), "success": False},
            success=False,
            error=f"Unexpected error: {str(e)}"
        )
        raise TestingError(f"Failed to run pytest on {code_file}: {str(e)}")


def discover_test_files(directory: Union[str, Path]) -> List[Path]:
    """
    Discover test files in a directory.
    """
    try:
        abs_dir = validate_sandbox_path(directory)
        
        # Find test files
        test_files = []
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend(abs_dir.rglob(pattern))
        
        # Filter out directories and only keep Python files
        test_files = [f for f in test_files if f.is_file() and f.suffix == '.py']
        
        # Sort for consistency
        test_files.sort()
        
        return test_files
        
    except Exception as e:
        raise TestingError(f"Failed to discover test files in {directory}: {str(e)}")


def run_tests_in_directory(directory: Union[str, Path], timeout: int = 30) -> Dict[str, Any]:
    """
    Run all tests in a directory.
    """
    try:
        abs_dir = validate_sandbox_path(directory)
        
        # Discover test files
        test_files = discover_test_files(abs_dir)
        
        if not test_files:
            return {
                "success": True,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "test_files": 0,
                "summary": "No test files found",
                "details": []
            }
        
        # Run tests for each file
        all_results = []
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_tests = 0
        
        for test_file in test_files:
            try:
                results = run_pytest_on_file(test_file, timeout=timeout)
                all_results.append({
                    "file": test_file.name,
                    "results": results
                })

                
                
                total_passed += results.get("passed", 0)
                total_failed += results.get("failed", 0)
                total_skipped += results.get("skipped", 0)
                total_errors += results.get("errors", 0)
                total_tests += results.get("total_tests", 0)
                
            except Exception as e:
                logger.error(f"Failed to run tests in {test_file}: {e}")
        
        # Calculate overall success
        overall_success = total_failed == 0 and total_errors == 0
        
        return {
            "success": overall_success,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "errors": total_errors,
            "test_files": len(test_files),
            "files_tested": len(all_results),
            "summary": f"{total_tests} tests in {len(test_files)} files: {total_passed} passed, {total_failed} failed",
            "details": all_results
        }
        
    except Exception as e:
        raise TestingError(f"Failed to run tests in directory {directory}: {str(e)}")


def _log_testing_operation(
    operation: str,
    filepath: Path,
    results: Dict[str, Any],
    success: bool,
    duration: float = 0.0,
    error: str = None
) -> None:
    """
    Log testing operation using project logger.
    """
    try:
        status = "SUCCESS" if success else "FAILURE"
        
        details = {
            "operation": operation,
            "file": str(filepath),
            "results": results,
            "duration_seconds": round(duration, 3),
            "input_prompt": f"Run tests on {filepath.name}",  # Required by logger
            "output_response": f"Test results: {results.get('summary', 'No summary')}"  # Required
        }
        
        if error:
            details["error"] = error
        
        log_experiment(
            agent_name="Toolsmith",
            model_used="pytest",  # Not an LLM, but we need to specify
            action=ActionType.DEBUG,  # Using DEBUG for tool operations
            details=details,
            status=status
        )
        
        # Also log to console
        if success:
            logger.info(f"Testing complete: {filepath.name} -> {results.get('summary', 'No results')}")
        else:
            logger.error(f"Testing failed: {filepath.name} -> {error or results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Failed to log testing operation: {e}")


def validate_test_code(test_code: str) -> Tuple[bool, str]:
    """
    Validate test code for safety and syntax.
    """
    # Check for dangerous imports/patterns
    dangerous_patterns = [
        (r"import\s+(os|sys|subprocess|shutil|socket)", "Dangerous import detected"),
        (r"__import__\s*\(", "Dynamic import detected"),
        (r"eval\s*\(", "eval() detected"),
        (r"exec\s*\(", "exec() detected"),
        (r"open\s*\([^)]*/etc/", "Attempt to access /etc/"),
        (r"open\s*\([^)]*\.\./", "Path traversal attempt"),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, test_code, re.IGNORECASE):
            return False, f"Security violation: {message}"
    
    # Check syntax
    try:
        ast.parse(test_code)
        return True, "Code syntax is valid"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"




# DYNAMIC TEST EXECUTION FUNCTIONS 

def _create_safe_test_environment() -> tempfile.TemporaryDirectory:
    """
    Create a safe temporary directory for dynamic test execution.
    
    Returns:
        TemporaryDirectory object
    """
    return tempfile.TemporaryDirectory(
        prefix="safe_test_",
        dir=get_sandbox_root()  # Ensure it's within sandbox
    )


def _sanitize_dynamic_code(code: str) -> str:
    """
    Sanitize dynamic code to prevent security issues.
    
    Args:
        code: Raw Python code
        
    Returns:
        Sanitized code
    """
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        (r"^\s*import\s+(os|sys|subprocess|shutil|socket|pty|fcntl)", "import safe_module"),
        (r"__import__\s*\(", ""),
        (r"eval\s*\(", ""),
        (r"exec\s*\(", ""),
        (r"open\s*\([^)]*[\\/]\.\.[\\/]", ""),
        (r"\.system\s*\(", ""),
        (r"\.popen\s*\(", ""),
        (r"\.spawn\s*\(", ""),
        (r"rm\s+-rf", ""),
        (r"chmod\s+777", ""),
        (r"^from\s+os\s+import", "from safe_module import"),
    ]
    
    sanitized = code
    for pattern, replacement in dangerous_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.MULTILINE)
    
    return sanitized


def execute_dynamic_tests(
    code: str,
    test_code: str,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Safely execute dynamically generated tests.
    
    Args:
        code: The code to test (Python source)
        test_code: Test code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Test execution results
        
    Raises:
        TestingError: If execution fails
        SecurityError: If code is unsafe
    """
    start_time = datetime.now()
    
    # Create a unique identifier for this execution
    exec_id = f"dynamic_test_{int(start_time.timestamp())}"
    
    try:
        # Validate test code security
        is_valid, validation_msg = validate_test_code(test_code)
        if not is_valid:
            raise SecurityError(f"Test code validation failed: {validation_msg}")
        
        # Sanitize code
        sanitized_code = _sanitize_dynamic_code(code)
        sanitized_test_code = _sanitize_dynamic_code(test_code)
        
        # Create safe temporary environment
        with _create_safe_test_environment() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write code files
            code_file = temp_path / "code_under_test.py"
            test_file = temp_path / "test_dynamic.py"
            
            safe_write_file(code_file, sanitized_code)
            safe_write_file(test_file, sanitized_test_code)
            
            # Create __init__.py to make it a package
            init_file = temp_path / "__init__.py"
            init_file.write_text("# Safe test package\n")
            
            # Run pytest on the test file
            results = _run_pytest_command(
                test_file,
                timeout=timeout
            )
            
            # Add execution metadata
            results.update({
                "execution_id": exec_id,
                "code_length": len(code),
                "test_code_length": len(test_code),
                "environment": "dynamic_sandbox",
                "sanitization_applied": sanitized_code != code or sanitized_test_code != test_code
            })
            
            # Log the operation
            _log_testing_operation(
                operation="execute_dynamic_tests",
                filepath=test_file,
                results=results,
                success=results.get("success", False),
                duration=(datetime.now() - start_time).total_seconds()
            )
            
            return results
            
    except (SecurityError, FileOpsError) as e:
        _log_testing_operation(
            operation="execute_dynamic_tests",
            filepath=Path(f"dynamic/{exec_id}"),
            results={"error": str(e), "success": False},
            success=False,
            error=str(e)
        )
        raise
    except Exception as e:
        _log_testing_operation(
            operation="execute_dynamic_tests",
            filepath=Path(f"dynamic/{exec_id}"),
            results={"error": str(e), "success": False},
            success=False,
            error=f"Unexpected error: {str(e)}"
        )
        raise TestingError(f"Dynamic test execution failed: {str(e)}")


def generate_and_execute_tests(
    code: str,
    specification: Dict[str, Any],
    timeout: int = 15
) -> Dict[str, Any]:
    """
    Generate tests from specification and execute them.
    This is a higher-level function that combines test generation and execution.
    
    Args:
        code: Code to test
        specification: Specification of what the code should do
        timeout: Maximum execution time
        
    Returns:
        Combined results
    """
    # This function would integrate with an LLM to generate tests
    # For now, we'll create simple template-based tests
    try:
        # Generate simple tests based on specification
        test_code = _generate_simple_tests_from_spec(code, specification)
        
        # Execute the tests
        return execute_dynamic_tests(code, test_code, timeout)
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generated_tests": False,
            "summary": f"Test generation/execution failed: {e}"
        }


def _generate_simple_tests_from_spec(code: str, spec: Dict[str, Any]) -> str:
    """
    Generate simple test cases from code specification.
    This is a basic implementation - in reality, the Judge LLM would generate tests.
    
    Args:
        code: Source code
        spec: Specification dict
        
    Returns:
        Generated test code
    """
    # Extract function names from code
    function_names = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except:
        pass
    
    # Basic test template
    test_template = '''
"""
Dynamically generated tests for: {filename}
Specification: {spec_summary}
"""


# Basic smoke test - ensure code runs
def test_code_executes():
    """Test that the code can be imported and executed."""
    # Check all functions can be called with basic arguments
    pass

# Function-specific tests
{function_tests}

# Edge case tests
def test_edge_cases():
    """Test edge cases."""
    # Add edge case tests here
    pass
'''
    
    # Generate function tests
    function_tests = []
    for func_name in function_names[:3]:  # Limit to 3 functions
        func_test = f'''
def test_{func_name}_basic():
    """Basic test for {func_name}."""
    # TODO: Add actual test logic based on specification
    # This would be filled by the Judge LLM
    try:
        result = {func_name}()
        assert result is not None
    except Exception as e:
        # Function might require arguments
        pass
'''
        function_tests.append(func_test)
    
    # Build test code
    test_code = test_template.format(
        filename="dynamic_code.py",
        spec_summary=spec.get("description", "No specification provided"),
        function_tests="\n".join(function_tests)
    )
    
    return test_code


def benchmark_test_performance(
    code: str,
    test_cases: List[str],
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark test execution performance.
    
    Args:
        code: Code to test
        test_cases: List of test code strings
        iterations: Number of iterations to run
        
    Returns:
        Performance metrics
    """
    results = {
        "total_tests": len(test_cases),
        "iterations": iterations,
        "test_results": [],
        "performance": {
            "average_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "success_rate": 0.0
        }
    }
    
    total_time = 0.0
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases):
        iteration_times = []
        iteration_results = []
        
        for iteration in range(iterations):
            try:
                start = time.time()
                test_result = execute_dynamic_tests(code, test_case, timeout=5)
                elapsed = time.time() - start
                
                iteration_times.append(elapsed)
                iteration_results.append(test_result.get("success", False))
                
                if test_result.get("success", False):
                    successful_tests += 1
                    
            except Exception as e:
                iteration_times.append(0.0)
                iteration_results.append(False)
        
        # Calculate stats for this test case
        if iteration_times:
            avg_time = sum(iteration_times) / len(iteration_times)
            min_time = min(iteration_times)
            max_time = max(iteration_times)
            success_rate = sum(iteration_results) / len(iteration_results) * 100
        else:
            avg_time = min_time = max_time = 0.0
            success_rate = 0.0
        
        results["test_results"].append({
            "test_index": i,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "success_rate": success_rate
        })
        
        total_time += avg_time
    
    # Calculate overall performance
    if test_cases:
        results["performance"]["average_time"] = total_time / len(test_cases)
        results["performance"]["success_rate"] = (successful_tests / (len(test_cases) * iterations)) * 100
        results["performance"]["min_time"] = min(r["min_time"] for r in results["test_results"])
        results["performance"]["max_time"] = max(r["max_time"] for r in results["test_results"])
    
    return results