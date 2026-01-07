"""Test analysis tools (pylint integration)"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Ensure the project root is on sys.path so `import src...` works.
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add src directly as a fallback
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.tools.analysis import (
    run_pylint_analysis,
    analyze_directory,
    compare_code_quality,
    AnalysisError,
    _calculate_pylint_score,
    _generate_recommendations,
)
from src.tools.security import get_sandbox_root

# Setup
SANDBOX = get_sandbox_root()


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for testing"""
    test_dir = SANDBOX / "test_analysis"
    test_dir.mkdir(exist_ok=True)
    
    # Good Python file
    good_code = '''
"""Sample module with good practices."""
import os
import sys

def calculate_average(numbers):
    """Calculate average of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        """Add to value."""
        self.value += x
        return self.value
'''
    
    good_file = test_dir / "good_code.py"
    good_file.write_text(good_code)
    
    # Bad Python file
    bad_code = '''
import os, sys  # Multiple imports on one line

x=5  # No spaces
y = 10

def bad_func():  # Missing docstring
    unused_var = 5
    return x+y  # Missing spaces
    
class badClass:  # Wrong naming
    def __init__(self):
        self.x=1
'''
    
    bad_file = test_dir / "bad_code.py"
    bad_file.write_text(bad_code)
    
    # Empty directory for testing
    empty_dir = test_dir / "empty"
    empty_dir.mkdir(exist_ok=True)
    
    yield test_dir


def test_calculate_pylint_score():
    """Test score calculation from issues"""
    # No issues -> perfect score
    score, counts = _calculate_pylint_score([])
    assert score == 10.0
    assert counts["total"] == 0
    
    # Mock issues
    issues = [
        {"type": "error", "message": "Syntax error"},
        {"type": "warning", "message": "Unused import"},
        {"type": "convention", "message": "Missing docstring"},
        {"type": "refactor", "message": "Simplify expression"},
    ]
    
    score, counts = _calculate_pylint_score(issues)
    assert score < 10.0  
    assert counts["total"] == 4
    assert counts["error"] == 1
    assert counts["warning"] == 1


def test_generate_recommendations():
    """Test recommendation generation"""
    # No issues
    recs = _generate_recommendations([], 9.5)
    assert "excellent" in recs[0].lower()
    
    # With issues
    issues = [
        {"type": "error", "symbol": "syntax-error", "message": "Invalid syntax", "line": 1},
        {"type": "warning", "symbol": "unused-import", "message": "Unused import", "line": 2},
    ]
    
    recs = _generate_recommendations(issues, 5.0)
    assert len(recs) > 0
    assert any("critical" in r.lower() for r in recs)


@patch('src.tools.analysis._run_pylint_command')
def test_run_pylint_analysis_success(mock_pylint, sample_python_file):
    """Test successful pylint analysis"""
    test_dir = sample_python_file
    test_file = test_dir / "good_code.py"
    
    # Mock pylint response
    mock_issues = [
        {"type": "convention", "symbol": "missing-docstring", "line": 10, "message": "Missing docstring"},
        {"type": "warning", "symbol": "unused-argument", "line": 15, "message": "Unused argument 'x'"},
    ]
    
    mock_pylint.return_value = {
        "issues": mock_issues,
        "errors": [],
        "return_code": 0,
        "command": "pylint ..."
    }
    
    # Run analysis
    result = run_pylint_analysis(test_file)
    
    # Verify structure
    assert "score" in result
    assert "issues" in result
    assert "issue_counts" in result
    assert "recommendations" in result
    assert "file_info" in result
    
    # Score should be calculated
    assert 0 <= result["score"] <= 10
    assert result["issue_counts"]["total"] == 2
    
    # Verify mock was called
    mock_pylint.assert_called_once()


@patch('src.tools.analysis._run_pylint_command')
def test_run_pylint_analysis_failure(mock_pylint, sample_python_file):
    """Test analysis failure"""
    test_dir = sample_python_file
    test_file = test_dir / "bad_code.py"
    
    # Mock pylint failure
    mock_pylint.side_effect = AnalysisError("Pylint failed")
    
    with pytest.raises(AnalysisError):
        run_pylint_analysis(test_file)


def test_run_pylint_analysis_security():
    """Test security in analysis"""
    with pytest.raises(Exception):  # Should be SecurityError or AnalysisError
        run_pylint_analysis("/etc/passwd")


@patch('src.tools.analysis.run_pylint_analysis')
def test_analyze_directory(mock_analysis, sample_python_file):
    """Test directory analysis"""
    test_dir = sample_python_file
    
    # Mock individual file analyses
    mock_analysis.side_effect = [
        {"score": 8.5, "issue_counts": {"total": 2}},
        {"score": 6.0, "issue_counts": {"total": 5}},
    ]
    
    result = analyze_directory(test_dir)
    
    assert "total_files" in result
    assert "average_score" in result
    assert "file_analyses" in result
    
    # Should have analyzed 2 files
    assert len(result["file_analyses"]) == 2
    assert result["average_score"] == 7.25  # (8.5 + 6.0) / 2


def test_analyze_empty_directory(sample_python_file):
    """Test analyzing empty directory"""
    test_dir = sample_python_file
    empty_dir = test_dir / "empty"
    
    result = analyze_directory(empty_dir)
    
    assert result["total_files"] == 0
    assert result["average_score"] == 0.0
    assert result["file_analyses"] == []


@patch('src.tools.analysis.run_pylint_analysis')
def test_compare_code_quality(mock_analysis):
    """Test quality comparison"""
    # Mock analyses
    mock_analysis.side_effect = [
        {"score": 5.0, "issue_counts": {"total": 10}},  # Original
        {"score": 8.0, "issue_counts": {"total": 3}},   # Improved
    ]
    
    result = compare_code_quality("old.py", "new.py")
    
    assert result["original_score"] == 5.0
    assert result["new_score"] == 8.0
    assert result["improvement"] == 3.0
    assert result["issues_reduced"] == 7


@patch('src.tools.analysis.log_experiment')
def test_analysis_logging(mock_logger, sample_python_file):
    """Test that analysis is logged"""
    test_dir = sample_python_file
    test_file = test_dir / "good_code.py"
    
    # Mock pylint
    with patch('src.tools.analysis._run_pylint_command') as mock_pylint:
        mock_pylint.return_value = {
            "issues": [],
            "errors": [],
            "return_code": 0,
            "command": "pylint"
        }
        
        run_pylint_analysis(test_file)
        
        # Verify logging
        assert mock_logger.called
        call_args = mock_logger.call_args
        assert call_args[1]['agent_name'] == "Toolsmith"
        assert call_args[1]['action'].value == "CODE_ANALYSIS"


def test_real_pylint_analysis(sample_python_file):
    """Integration test with real pylint (requires pylint installed)"""
    test_dir = sample_python_file
    
    # Skip if pylint not available
    try:
        import pylint
    except ImportError:
        pytest.skip("pylint not installed")
    
    # Test on good code
    good_file = test_dir / "good_code.py"
    result = run_pylint_analysis(good_file)
    
    # Verify structure
    assert isinstance(result, dict)
    assert "score" in result
    assert "issues" in result
    
    # Good code should have decent score
    # Note: This might vary based on pylint version
    assert result["score"] > 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])