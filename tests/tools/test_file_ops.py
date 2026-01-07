"""Test file operations with logging"""
import sys
import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the project root is on sys.path so `import src...` works.
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add src directly as a fallback
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.tools.file_ops import (
    safe_read_file,
    safe_write_file,
    safe_list_files,
    safe_read_json,
    safe_write_json,
    FileOpsError,
    initialize_sandbox,
)
from src.tools.security import get_sandbox_root, SecurityError

# Setup
SANDBOX = get_sandbox_root()


@pytest.fixture
def setup_sandbox():
    """Create test files in sandbox"""
    test_dir = SANDBOX / "test_file_ops"
    test_dir.mkdir(exist_ok=True)
    
    # Create test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Hello, World!")
    
    # Create subdirectory
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "nested.txt").write_text("Nested content")
    
    yield test_dir


def test_safe_read_file(setup_sandbox):
    """Test reading files"""
    test_dir = setup_sandbox
    test_file = test_dir / "test.txt"
    
    content = safe_read_file(test_file)
    assert content == "Hello, World!"
    
    # Test with string path
    content = safe_read_file(str(test_file))
    assert content == "Hello, World!"


def test_safe_read_file_security():
    """Test security in read operations"""
    with pytest.raises(SecurityError):
        safe_read_file("/etc/passwd")
    
    with pytest.raises(SecurityError):
        safe_read_file("../../../escape.py")


def test_safe_read_file_errors(setup_sandbox):
    """Test error cases"""
    test_dir = setup_sandbox
    
    # Non-existent file
    with pytest.raises(FileOpsError):
        safe_read_file(test_dir / "nonexistent.txt")
    
    # Directory instead of file
    with pytest.raises(FileOpsError):
        safe_read_file(test_dir)


def test_safe_write_file(setup_sandbox):
    """Test writing files"""
    test_dir = setup_sandbox
    new_file = test_dir / "new.txt"
    
    # Write new file
    result = safe_write_file(new_file, "New content")
    assert result is True
    assert new_file.read_text() == "New content"
    
    # Overwrite existing
    result = safe_write_file(new_file, "Updated content")
    assert result is True
    assert new_file.read_text() == "Updated content"


def test_safe_write_file_no_overwrite(setup_sandbox):
    """Test write without overwrite"""
    test_dir = setup_sandbox
    existing_file = test_dir / "existing.txt"
    existing_file.write_text("Original")
    
    with pytest.raises(FileOpsError):
        safe_write_file(existing_file, "New", overwrite=False)


def test_safe_write_file_creates_directories(setup_sandbox):
    """Test that parent directories are created"""
    test_dir = setup_sandbox
    deep_file = test_dir / "deep" / "nested" / "file.txt"
    
    result = safe_write_file(deep_file, "Deep content")
    assert result is True
    assert deep_file.exists()
    assert deep_file.read_text() == "Deep content"


def test_safe_list_files(setup_sandbox):
    """Test listing files"""
    test_dir = setup_sandbox
    
    files = safe_list_files(test_dir)
    file_names = [f.name for f in files]
    
    assert "test.txt" in file_names
    
    # Test recursive
    all_files = safe_list_files(test_dir, recursive=True)
    all_names = [f.name for f in all_files]
    assert "test.txt" in all_names
    assert "nested.txt" in all_names


def test_safe_list_files_with_pattern(setup_sandbox):
    """Test listing with pattern"""
    test_dir = setup_sandbox
    
    # Create more test files
    (test_dir / "script.py").write_text("import os")
    (test_dir / "data.json").write_text('{"key": "value"}')
    
    # List only Python files
    py_files = safe_list_files(test_dir, pattern="*.py")
    assert any(f.name == "script.py" for f in py_files)
    assert not any(f.name == "data.json" for f in py_files)


def test_safe_json_operations(setup_sandbox):
    """Test JSON read/write"""
    test_dir = setup_sandbox
    json_file = test_dir / "data.json"
    
    data = {"name": "test", "values": [1, 2, 3]}
    
    # Write JSON
    result = safe_write_json(json_file, data)
    assert result is True
    
    # Read JSON
    loaded = safe_read_json(json_file)
    assert loaded == data
    
    # Verify file content
    content = json_file.read_text()
    parsed = json.loads(content)
    assert parsed == data


def test_safe_json_errors(setup_sandbox):
    """Test JSON error handling"""
    test_dir = setup_sandbox
    
    # Invalid JSON file
    bad_json = test_dir / "bad.json"
    bad_json.write_text("{invalid json}")
    
    with pytest.raises(FileOpsError):
        safe_read_json(bad_json)


def test_logging_integration(setup_sandbox):
    """Test that operations are logged"""
    test_dir = setup_sandbox
    test_file = test_dir / "log_test.txt"
    
    # Mock the logger to verify calls
    with patch('src.tools.file_ops.log_experiment') as mock_log:
        safe_write_file(test_file, "Test content")
        
        # Verify log_experiment was called
        assert mock_log.called
        
        # Check call arguments
        call_args = mock_log.call_args
        assert call_args[1]['agent_name'] == "Toolsmith"
        assert call_args[1]['status'] == "SUCCESS"


def test_initialize_sandbox():
    """Test sandbox initialization"""
    # Remove sandbox if exists (be careful!)
    # if SANDBOX.exists():
    #     shutil.rmtree(SANDBOX)
    
    initialize_sandbox()
    assert SANDBOX.exists()
    assert SANDBOX.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])