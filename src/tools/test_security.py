"""Test security layer with malicious and safe paths"""
import os
import sys
import pytest
from pathlib import Path
from security import (
    is_within_sandbox,
    validate_sandbox_path,
    sanitize_filename,
    SecurityError,
    get_sandbox_root
)

# Setup
SANDBOX = get_sandbox_root()


def test_safe_paths():
    """Test paths that should be allowed"""
    safe_paths = [
        SANDBOX / "test.py",
        SANDBOX / "subdir" / "test.py",
        str(SANDBOX / "test.py"),
        "./sandbox/test.py",  # Relative path
    ]
    
    for path in safe_paths:
        assert is_within_sandbox(path), f"Safe path rejected: {path}"


def test_malicious_paths():
    """Test path traversal attempts that should be blocked"""
    malicious_paths = [
        "/etc/passwd",  # System file
        "C:\\Windows\\System32",  # Windows system
        "../../../etc/passwd",  # Path traversal
        f"{SANDBOX}/../config.py",  # Parent directory
        f"{SANDBOX}/test/../../..",  # Complex traversal
        "/tmp/test.py",  # Outside sandbox
        "~/.ssh/id_rsa",  # Home directory
    ]
    
    for path in malicious_paths:
        assert not is_within_sandbox(path), f"Malicious path allowed: {path}"


def test_validate_sandbox_path():
    """Test path validation"""
    # Should work
    valid = validate_sandbox_path(SANDBOX / "test.py")
    assert str(valid).endswith("sandbox/test.py")
    
    # Should fail
    with pytest.raises(SecurityError):
        validate_sandbox_path("/etc/passwd")


def test_sanitize_filename():
    """Test filename sanitization"""
    test_cases = [
        ("../../../etc/passwd", "etcpasswd"),
        ("test.py", "test.py"),
        ("malicious\x00file.py", "maliciousfile.py"),
        ("  .hidden.py  ", "hidden.py"),
        ("test; rm -rf /;", "test rm -rf"),
    ]
    
    for input_name, expected in test_cases:
        result = sanitize_filename(input_name)
        assert result == expected, f"Failed: {input_name} -> {result}"


def test_edge_cases():
    """Test edge cases"""
    # Empty sandbox path
    assert is_within_sandbox(SANDBOX)
    
    # Same path as sandbox
    assert is_within_sandbox(str(SANDBOX))
    


def test_windows_paths():
    """Test Windows-style paths (cross-platform compatibility)"""
    if os.name == 'nt':  # Windows
        # Convert to Windows path for testing
        windows_sandbox = str(SANDBOX).replace("/", "\\")
        assert is_within_sandbox(windows_sandbox + "\\test.py")
        
        # Different drive should fail
        assert not is_within_sandbox("D:\\test.py")


def test_logging_security_violation(caplog):
    """Test that security violations are logged"""
    import logging
    caplog.set_level(logging.ERROR)
    
    # This should log an error
    with pytest.raises(SecurityError):
        validate_sandbox_path("/etc/passwd")
    
    # Check error was logged
    assert "SECURITY VIOLATION" in caplog.text


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])