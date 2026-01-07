#!/usr/bin/env python3
"""Integration test for file operations"""
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

from src.tools import (
    initialize_sandbox,
    safe_write_file,
    safe_read_file,
    safe_list_files,
    safe_write_json,
    safe_read_json,
    get_sandbox_root,
)

def run_integration_test():
    """Run full integration test"""
    print("File Operations Integration Test")
    print("=" * 60)
    
    # Initialize
    initialize_sandbox()
    sandbox = get_sandbox_root()
    test_dir = sandbox / "integration_test"
    
    # Clean previous test
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_dir.mkdir()
    
    print("\n1 Testing file write/read...")
    
    # Test 1: Basic file operations
    test_file = test_dir / "test.txt"
    test_content = "Hello from Toolsmith!"
    
    safe_write_file(test_file, test_content)
    print(f"   Wrote: {test_file}")
    
    read_content = safe_read_file(test_file)
    assert read_content == test_content
    print(f"   Read: {len(read_content)} chars match")
    
    # Test 2: JSON operations
    print("\n2 Testing JSON operations...")
    
    json_file = test_dir / "data.json"
    json_data = {
        "project": "Refactoring Swarm",
        "toolsmith": "You",
        "files": ["auditor.py", "fixer.py", "judge.py"],
        "status": "in_progress"
    }
    
    safe_write_json(json_file, json_data)
    print(f"   Wrote JSON: {json_file}")
    
    loaded_data = safe_read_json(json_file)
    assert loaded_data == json_data
    print(f"   Read JSON: {len(loaded_data)} keys match")
    
    # Test 3: List files
    print("\n3 Testing file listing...")

    # Create more files
    safe_write_file(test_dir / "script.py", "print('hello')")
    safe_write_file(test_dir / "config.yaml", "key: value")
    
    all_files = safe_list_files(test_dir)
    print(f"   Found {len(all_files)} files in directory:")
    for f in all_files:
        print(f"     - {f.name}")
    
    # Test 4: Pattern filtering
    print("\n4 Testing pattern filtering...")

    py_files = safe_list_files(test_dir, pattern="*.py")
    print(f"   Found {len(py_files)} Python files")
    
    # Test 5: Security validation
    print("\n5 Testing security...")

    try:
        # This should fail
        safe_write_file("/tmp/evil.py", "malicious")
        print("   ERROR: Should have blocked outside sandbox!")
    except Exception as e:
        print(f"   Correctly blocked: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print(" All integration tests passed!")
    
    # Show log file (if exists)
    log_file = Path("logs") / "experiment_data.json"
    if log_file.exists():
        print(f"\n Log entries created: {log_file}")
        # Show last few entries
        try:
            import json
            with open(log_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"   Total log entries: {len(data)}")
                    # Show Toolsmith entries
                    tool_entries = [e for e in data[-5:] if e.get('agent_name') == 'Toolsmith']
                    for entry in tool_entries:
                        print(f"   - {entry.get('timestamp', '')}: {entry.get('details', {}).get('operation', '')}")
        except:
            pass

if __name__ == "__main__":
    run_integration_test()