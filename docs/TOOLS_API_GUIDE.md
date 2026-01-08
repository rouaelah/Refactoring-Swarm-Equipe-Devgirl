# Tools API - Complete Guide for Developers

## Overview
This document describes the complete internal API for the Refactoring Swarm project. All tools are sandbox-secure, logged, and ready for agent integration.

## Quick Start

```python
# Basic import
from src.tools import (
    safe_read_file, safe_write_file,  # File operations
    run_pylint_analysis,              # Code analysis (Auditor)
    run_pytest_on_file,               # Test execution (Judge)
    execute_dynamic_tests,            # Dynamic testing (Judge)
)

# Initialize sandbox (automatically done on import)
# All operations are restricted to the 'sandbox/' directory

Agent-Specific Imports
Auditor Agent
python

from src.tools import (
    safe_read_file,
    safe_list_files,
    run_pylint_analysis,
    analyze_directory,
    compare_code_quality,
)

Fixer Agent
python

from src.tools import (
    safe_read_file,
    safe_write_file,
    safe_read_json,
    safe_write_json,
)

Judge Agent
python

from src.tools import (
    safe_read_file,
    run_pytest_on_file,
    discover_test_files,
    run_tests_in_directory,
    validate_test_code,
    execute_dynamic_tests,
    generate_and_execute_tests,
)

Security Features

    All file operations are restricted to sandbox/ directory

    Path traversal attempts are blocked (../../../etc/passwd)

    Dangerous imports are sanitized (os, sys, subprocess)

    Dynamic code execution has timeouts and sandboxing

Logging Integration

Every tool call is automatically logged to logs/experiment_data.json with:

    Timestamp

    Agent name

    Action type

    Input/Output (required for grading)

    Success/Failure status

File Operations
Reading Files
python

content = safe_read_file("sandbox/code.py")
# Returns: string content
# Raises: SecurityError, FileOpsError

Writing Files
python

success = safe_write_file("sandbox/fixed.py", new_code)
# Returns: True on success
# Raises: SecurityError, FileOpsError

Listing Files
python

files = safe_list_files("sandbox/", pattern="*.py", recursive=True)
# Returns: List[Path] objects

Code Analysis (Auditor)
Basic Analysis
python

results = run_pylint_analysis("sandbox/code.py")
# Returns: {
#   "score": 7.5,                    # 0-10 quality score
#   "issues": [...],                 # Detailed issues
#   "issue_counts": {"error": 2, ...},
#   "recommendations": ["Fix ..."],  # Actionable suggestions
# }

Directory Analysis
python

summary = analyze_directory("sandbox/")
# Returns: Summary of all Python files in directory

Quality Comparison
python

comparison = compare_code_quality("old.py", "new.py")
# Returns: Improvement metrics for grading

Testing (Judge)
Running Existing Tests
python

results = run_pytest_on_file("sandbox/code.py", timeout=30)
# Returns: {
#   "tool_success": True,    # Did pytest run?
#   "tests_passed": False,   # Did all tests pass?
#   "total_tests": 5,
#   "passed": 4,
#   "failed": 1,
#   "summary": "...",
# }

Dynamic Test Execution
python

results = execute_dynamic_tests(
    code="def add(a,b): return a+b",
    test_code="def test_add(): assert add(2,3)==5",
    timeout=10
)
# Security: Blocks dangerous code, has timeout protection

Test Generation & Execution
python

results = generate_and_execute_tests(
    code=python_code,
    specification={"description": "Function calculates average"},
    timeout=15
)
# Generates tests based on spec, then executes them

Error Handling

All functions raise specific exceptions:
python

try:
    content = safe_read_file("/etc/passwd")
except SecurityError as e:
    print(f"Security violation: {e}")
except FileOpsError as e:
    print(f"File operation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

Example Workflows
Auditor Workflow
python

# 1. Read code
code = safe_read_file("sandbox/buggy.py")

# 2. Analyze quality
analysis = run_pylint_analysis("sandbox/buggy.py")

# 3. Create refactoring plan based on analysis
plan = create_refactoring_plan(analysis)

Judge Workflow
python

# 1. Run existing tests
test_results = run_pytest_on_file("sandbox/fixed.py")

# 2. If tests fail, maybe generate new tests
if not test_results["tests_passed"]:
    new_tests = generate_tests_based_on_failures(test_results)
    results = execute_dynamic_tests(code, new_tests)
    
# 3. Decide: loop or success

🔧 Integration with LLM Prompts

When writing agent prompts, include tool usage:
python

prompt_template = """
You are an Auditor agent. You have access to these tools:

1. safe_read_file(path) - Read a file
2. run_pylint_analysis(path) - Get code quality metrics

Your task:
1. Use safe_read_file() to read {filepath}
2. Use run_pylint_analysis() to analyze it
3. Create a refactoring plan based on the results
"""