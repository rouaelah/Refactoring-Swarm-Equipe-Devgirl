"""
Tools package - Internal API for agents.
This module provides all the tools needed by the Refactoring Swarm agents.
All tools are sandbox-secure, logged, and ready for production use.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Toolsmith Team"
__description__ = "Internal API tools for Auditor, Fixer, and Judge agents"

# Security module exports
from .security import (
    SecurityError,
    is_within_sandbox,
    validate_sandbox_path,
    sanitize_filename,
    get_sandbox_root,
    create_sandbox_if_not_exists,
)

# File operations exports
from .file_ops import (
    FileOpsError,
    safe_read_file,
    safe_write_file,
    safe_list_files,
    safe_read_json,
    safe_write_json,
    initialize_sandbox,
)

# Analysis module exports (for Auditor)
from .analysis import (
    AnalysisError,
    run_pylint_analysis,
    analyze_directory,
    compare_code_quality,
)

# Testing module exports (for Judge)
from .testing import (
    TestingError,
    TestTimeoutError,
    run_pytest_on_file,
    discover_test_files,
    run_tests_in_directory,
    validate_test_code,
    execute_dynamic_tests,
    generate_and_execute_tests,
    benchmark_test_performance,
)

# Tool categories for easy reference
TOOL_CATEGORIES = {
    "security": [
        "SecurityError",
        "is_within_sandbox",
        "validate_sandbox_path",
        "sanitize_filename",
        "get_sandbox_root",
        "create_sandbox_if_not_exists",
    ],
    "file_ops": [
        "FileOpsError",
        "safe_read_file",
        "safe_write_file",
        "safe_list_files",
        "safe_read_json",
        "safe_write_json",
        "initialize_sandbox",
    ],
    "analysis": [
        "AnalysisError",
        "run_pylint_analysis",
        "analyze_directory",
        "compare_code_quality",
    ],
    "testing": [
        "TestingError",
        "TestTimeoutError",
        "run_pytest_on_file",
        "discover_test_files",
        "run_tests_in_directory",
        "validate_test_code",
        "execute_dynamic_tests",
        "generate_and_execute_tests",
        "benchmark_test_performance",
    ],
}

# Agent-specific tool mappings
AGENT_TOOLS = {
    "auditor": [
        "safe_read_file",
        "safe_list_files",
        "run_pylint_analysis",
        "analyze_directory",
        "compare_code_quality",
    ],
    "fixer": [
        "safe_read_file",
        "safe_write_file",
        "safe_read_json",
        "safe_write_json",
    ],
    "judge": [
        "safe_read_file",
        "run_pytest_on_file",
        "discover_test_files",
        "run_tests_in_directory",
        "validate_test_code",
        "execute_dynamic_tests",
        "generate_and_execute_tests",
    ],
}

# Complete export list
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__description__",
    "TOOL_CATEGORIES",
    "AGENT_TOOLS",
    
    # Security
    "SecurityError",
    "is_within_sandbox",
    "validate_sandbox_path",
    "sanitize_filename",
    "get_sandbox_root",
    "create_sandbox_if_not_exists",
    
    # File operations
    "FileOpsError",
    "safe_read_file",
    "safe_write_file",
    "safe_list_files",
    "safe_read_json",
    "safe_write_json",
    "initialize_sandbox",
    
    # Analysis
    "AnalysisError",
    "run_pylint_analysis",
    "analyze_directory",
    "compare_code_quality",
    
    # Testing
    "TestingError",
    "TestTimeoutError",
    "run_pytest_on_file",
    "discover_test_files",
    "run_tests_in_directory",
    "validate_test_code",
    "execute_dynamic_tests",
    "generate_and_execute_tests",
    "benchmark_test_performance",
]

# Initialize sandbox on import
initialize_sandbox()