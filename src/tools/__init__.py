"""
Tools package - Internal API for agents.
"""

from .security import (
    SecurityError,
    is_within_sandbox,
    validate_sandbox_path,
    sanitize_filename,
    get_sandbox_root,
    create_sandbox_if_not_exists,
)

from .file_ops import (
    FileOpsError,
    safe_read_file,
    safe_write_file,
    safe_list_files,
    safe_read_json,
    safe_write_json,
    initialize_sandbox,
)

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

__all__ = [
    # Security
    'SecurityError',
    'is_within_sandbox',
    'validate_sandbox_path',
    'sanitize_filename',
    'get_sandbox_root',
    'create_sandbox_if_not_exists',
    
    # File operations
    'FileOpsError',
    'safe_read_file',
    'safe_write_file',
    'safe_list_files',
    'safe_read_json',
    'safe_write_json',
    'initialize_sandbox',

    # Testing
    'TestingError',
    'TestTimeoutError',
    'run_pytest_on_file',
    'discover_test_files', 
    'run_tests_in_directory',
    'validate_test_code',
    'execute_dynamic_tests',
    'generate_and_execute_tests',
    'benchmark_test_performance',
]