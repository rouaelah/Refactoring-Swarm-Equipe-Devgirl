"""
Safe file operations with comprehensive logging.
All operations are restricted to sandbox directory.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from datetime import datetime

# Ensure the project root is on sys.path so `import src...` works.
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add src directly as a fallback
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the security module
from src.tools.security import validate_sandbox_path, SecurityError, create_sandbox_if_not_exists
from src.utils.logger import log_experiment, ActionType

logger = logging.getLogger(__name__)


class FileOpsError(Exception):
    """Custom exception for file operation errors"""
    pass


def _log_file_operation(
    operation: str,
    filepath: Path,
    success: bool,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> None:
    """
    Log file operation using the project's experiment logger.
    """
    try:
        status = "SUCCESS" if success else "FAILURE"
        
        log_details = {
            "operation": operation,
            "file_path": str(filepath),
            "timestamp": datetime.now().isoformat(),
            **(details or {})
        }
        
        if error:
            log_details["error"] = error
        
        # Log to experiment_data.json
        log_experiment(
            agent_name="Toolsmith",
            model_used="Toolsmith",  # We're not using LLM here
            action=ActionType.DEBUG,  # Using DEBUG for tool operations
            details=log_details,
            status=status
        )
        
        # Also log to console for debugging
        if success:
            logger.info(f"File operation {operation} succeeded: {filepath}")
        else:
            logger.error(f"File operation {operation} failed: {error}")
            
    except Exception as e:
        logger.error(f"Failed to log file operation: {e}")


def safe_read_file(filepath: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Safely read a file within sandbox.
    """
    start_time = datetime.now()
    
    try:
        # Validate path is within sandbox
        validated_path = validate_sandbox_path(filepath)
        
        # Check if file exists
        if not validated_path.exists():
            raise FileOpsError(f"File does not exist: {validated_path}")
        
        # Check if it's a file
        if not validated_path.is_file():
            raise FileOpsError(f"Path is not a file: {validated_path}")
        
        # Read file
        with open(validated_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Log successful operation
        _log_file_operation(
            operation="read",
            filepath=validated_path,
            success=True,
            details={
                "file_size": len(content),
                "encoding": encoding,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "input_prompt": f"Read file: {validated_path}",  # Required by logger
                "output_response": f"Successfully read {len(content)} characters"  # Required
            }
        )
        
        return content
        
    except (SecurityError, FileOpsError) as e:
        # These are expected errors
        _log_file_operation(
            operation="read",
            filepath=Path(filepath),
            success=False,
            error=str(e),
            details={
                "input_prompt": f"Read file: {filepath}",
                "output_response": f"Error: {str(e)}"
            }
        )
        raise
    except Exception as e:
        # Unexpected error
        _log_file_operation(
            operation="read",
            filepath=Path(filepath),
            success=False,
            error=f"Unexpected error: {str(e)}",
            details={
                "input_prompt": f"Read file: {filepath}",
                "output_response": f"Unexpected error: {str(e)}"
            }
        )
        raise FileOpsError(f"Failed to read file {filepath}: {str(e)}")


def safe_write_file(
    filepath: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    overwrite: bool = True
) -> bool:
    """
    Safely write content to a file within sandbox.
    """
    start_time = datetime.now()
    
    try:
        # Validate path is within sandbox
        validated_path = validate_sandbox_path(filepath)
        
        # Check if file exists and we shouldn't overwrite
        if validated_path.exists() and not overwrite:
            raise FileOpsError(f"File already exists and overwrite=False: {validated_path}")
        
        # Create parent directories if needed
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(validated_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        # Verify file was written
        if not validated_path.exists():
            raise FileOpsError(f"File not created after write: {validated_path}")
        
        # Log successful operation
        _log_file_operation(
            operation="write",
            filepath=validated_path,
            success=True,
            details={
                "file_size": len(content),
                "encoding": encoding,
                "overwrite": overwrite,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "input_prompt": f"Write to file: {validated_path}",
                "output_response": f"Successfully wrote {len(content)} characters"
            }
        )
        
        return True
        
    except (SecurityError, FileOpsError) as e:
        _log_file_operation(
            operation="write",
            filepath=Path(filepath),
            success=False,
            error=str(e),
            details={
                "input_prompt": f"Write to file: {filepath}",
                "output_response": f"Error: {str(e)}"
            }
        )
        raise
    except Exception as e:
        _log_file_operation(
            operation="write",
            filepath=Path(filepath),
            success=False,
            error=f"Unexpected error: {str(e)}",
            details={
                "input_prompt": f"Write to file: {filepath}",
                "output_response": f"Unexpected error: {str(e)}"
            }
        )
        raise FileOpsError(f"Failed to write file {filepath}: {str(e)}")


def safe_list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    Safely list files in a directory within sandbox.
    """
    start_time = datetime.now()
    
    try:
        # Validate directory is within sandbox
        validated_dir = validate_sandbox_path(directory)
        
        # Check if it's a directory
        if not validated_dir.is_dir():
            raise FileOpsError(f"Path is not a directory: {validated_dir}")
        
        # List files
        if recursive:
            files = list(validated_dir.rglob(pattern))
        else:
            files = list(validated_dir.glob(pattern))
        
        # Filter out directories (keep only files)
        files = [f for f in files if f.is_file()]
        
        # Log successful operation
        _log_file_operation(
            operation="list_files",
            filepath=validated_dir,
            success=True,
            details={
                "pattern": pattern,
                "recursive": recursive,
                "file_count": len(files),
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "input_prompt": f"List files in: {validated_dir}",
                "output_response": f"Found {len(files)} files"
            }
        )
        
        return files
        
    except (SecurityError, FileOpsError) as e:
        _log_file_operation(
            operation="list_files",
            filepath=Path(directory),
            success=False,
            error=str(e),
            details={
                "input_prompt": f"List files in: {directory}",
                "output_response": f"Error: {str(e)}"
            }
        )
        raise
    except Exception as e:
        _log_file_operation(
            operation="list_files",
            filepath=Path(directory),
            success=False,
            error=f"Unexpected error: {str(e)}",
            details={
                "input_prompt": f"List files in: {directory}",
                "output_response": f"Unexpected error: {str(e)}"
            }
        )
        raise FileOpsError(f"Failed to list files in {directory}: {str(e)}")


def safe_read_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely read and parse a JSON file within sandbox.
    """
    try:
        content = safe_read_file(filepath)
        
        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise FileOpsError(f"Invalid JSON in {filepath}: {str(e)}")
        
        return data
        
    except Exception as e:
        if not isinstance(e, (SecurityError, FileOpsError)):
            raise FileOpsError(f"Failed to read JSON file {filepath}: {str(e)}")
        raise


def safe_write_json(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2
) -> bool:
    """
    Safely write data as JSON to a file within sandbox.
    """
    try:
        # Convert to JSON
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        
        # Write file
        return safe_write_file(filepath, content)
        
    except Exception as e:
        if not isinstance(e, (SecurityError, FileOpsError)):
            raise FileOpsError(f"Failed to write JSON file {filepath}: {str(e)}")
        raise


def initialize_sandbox() -> None:
    """
    Initialize the sandbox directory.
    Should be called at system startup.
    """
    create_sandbox_if_not_exists()
    logger.info("File operations system initialized")