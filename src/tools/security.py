"""
Security module for sandbox operations.
Ensures all file operations stay within the sandbox directory.
"""
import os
import logging
from pathlib import Path, PurePath
from typing import Union, Tuple

# Get the project root directory 
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Sandbox directory 
SANDBOX_DIR = PROJECT_ROOT / "sandbox"

# Logger for security violations
logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security violations"""
    pass


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Convert any path to absolute normalized Path.
    Handles both string and Path objects.
    """
    try:
        return Path(path).expanduser().resolve()
    except Exception as e:
        raise SecurityError(f"Cannot normalize path {path}: {str(e)}")


def is_within_sandbox(filepath: Union[str, Path]) -> bool:
    """
    Check if a filepath is within the sandbox directory.
    """
    try:
        abs_path = normalize_path(filepath)
        
        # Check if path is within sandbox
        try:
            # Convert to string for commonpath
            sandbox_str = str(SANDBOX_DIR.absolute())
            path_str = str(abs_path)
            
            # Check if sandbox is prefix of path
            if not path_str.startswith(sandbox_str + os.sep) and path_str != sandbox_str:
                return False
            
            # Additional safety: ensure common path calculation works
            common = os.path.commonpath([sandbox_str, path_str])
            return common == sandbox_str
        except ValueError:
            return False
            
    except Exception as e:
        raise SecurityError(f"Security check failed for {filepath}: {str(e)}")


def validate_sandbox_path(filepath: Union[str, Path]) -> Path:
    """
    Validate a path is within sandbox, return normalized path.
    """
    abs_path = normalize_path(filepath)
    
    if not is_within_sandbox(abs_path):
        # Log the violation
        logger.error(f"SECURITY VIOLATION: Attempted access outside sandbox: {filepath}")
        raise SecurityError(
            f"Access denied. Path '{filepath}' is outside the sandbox directory. "
            f"Sandbox: {SANDBOX_DIR}"
        )
    
    return abs_path


def sanitize_filename(filename: str) -> str:
    """
    Remove dangerous characters from filename to prevent path traversal.
    """
    # Remove path traversal sequences
    sanitized = filename.replace("..", "").replace("//", "").replace("\\", "").replace("/", "")
    
    # Remove null bytes and other dangerous characters
    dangerous = ["\x00", "\r", "\n", "|", ";", "&", "`", "$", "(", ")", "<", ">"]
    for char in dangerous:
        sanitized = sanitized.replace(char, "")
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip(".")
    
    return sanitized if sanitized else "sanitized_file"


def create_sandbox_if_not_exists() -> None:
    """
    Ensure sandbox directory exists.
    Called at system startup.
    """
    SANDBOX_DIR.mkdir(exist_ok=True, parents=True)
    logger.info(f"Sandbox directory ready: {SANDBOX_DIR}")


def get_sandbox_root() -> Path:
    """Get the absolute path to sandbox root"""
    return SANDBOX_DIR