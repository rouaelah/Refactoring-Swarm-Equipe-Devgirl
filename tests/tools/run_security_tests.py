"""Quick test script for security layer"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure the project root is on sys.path so `import src...` works.
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add src directly as a fallback
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.tools.security import (
    is_within_sandbox,
    validate_sandbox_path,
    sanitize_filename,
    get_sandbox_root
)

def run_manual_tests():
    """Run manual tests without pytest"""
    sandbox = get_sandbox_root()
    
    print("Security Layer Tests")
    print("=" * 50)
    
    # Test 1: Safe paths
    print("\nSafe Paths (should be allowed):")
    safe_paths = [
        sandbox / "test.py",
        str(sandbox / "subdir" / "file.py"),
        "./sandbox/script.py",
    ]
    
    for path in safe_paths:
        result = is_within_sandbox(path)
        print(f"  {path} -> {'ALLOWED' if result else 'BLOCKED'}")
        assert result, f"Safe path blocked: {path}"
    
    # Test 2: Malicious paths
    print("\nMalicious Paths (should be blocked):")
    malicious_paths = [
        "/etc/passwd",
        "../../../etc/passwd",
        f"{sandbox}/../escape.py",
        "~/.ssh/id_rsa",
    ]
    
    for path in malicious_paths:
        result = is_within_sandbox(path)
        print(f"  {path:40} -> {'ALLOWED' if result else 'BLOCKED'}")
        assert not result, f"Malicious path allowed: {path}"
    
    # Test 3: Validation
    print("\nPath Validation:")
    try:
        validated = validate_sandbox_path(sandbox / "valid.py")
        print(f"  Valid path -> {validated}")
        
        validate_sandbox_path("/etc/passwd")
        print("  ERROR: Should have raised SecurityError!")
    except Exception as e:
        print(f"  Correctly blocked: {e}")
    
    # Test 4: Filename sanitization
    print("\nFilename Sanitization:")
    test_cases = [
        ("../../../bad.py", "bad.py"),
        ("test; ls;", "test ls"),
        ("normal.py", "normal.py"),
    ]
    
    for dirty, expected_clean in test_cases:
        clean = sanitize_filename(dirty)
        print(f"  '{dirty}' -> '{clean}'")
        assert clean == expected_clean
    
    print("\n" + "=" * 50)
    print("All manual tests passed!")

if __name__ == "__main__":
    run_manual_tests()