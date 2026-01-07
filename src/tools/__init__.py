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

from .analysis import (
    AnalysisError,
    run_pylint_analysis,
    analyze_directory,
    compare_code_quality,
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

    # Analysis
    'AnalysisError',
    'run_pylint_analysis',
    'analyze_directory',
    'compare_code_quality',
]