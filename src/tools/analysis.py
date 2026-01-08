"""
Analysis tools for code quality assessment.
Integrates with Pylint for static analysis.
"""
import json
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging

from .security import validate_sandbox_path, SecurityError
from .file_ops import safe_read_file, FileOpsError
from src.utils.logger import log_experiment, ActionType

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Custom exception for analysis failures"""
    pass


def _run_pylint_command(filepath: Path, extra_args: List[str] = None) -> Dict[str, Any]:
    """
    Run pylint on a file and return parsed results.
    
    Args:
        filepath: Path to Python file
        extra_args: Additional pylint arguments
        
    Returns:
        Dictionary with pylint results
    """
    if extra_args is None:
        extra_args = []
    
    # Build command
    cmd = [
        "pylint",
        "--output-format=json",  # JSON for easy parsing
        "--reports=n",           # No reports (we'll calculate score)
        "--exit-zero",           # Don't fail on linting issues
        *extra_args,
        str(filepath)
    ]
    
    try:
        logger.debug(f"Running pylint: {' '.join(cmd)}")
        
        # Run pylint with timeout (30 seconds max)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=filepath.parent
        )
        
        # Parse output
        if result.stdout.strip():
            try:
                issues = json.loads(result.stdout)
                if not isinstance(issues, list):
                    issues = []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse pylint JSON: {result.stdout[:100]}")
                issues = []
        else:
            issues = []
        
        # Check for errors
        errors = []
        if result.stderr:
            error_lines = result.stderr.strip().split('\n')
            errors = [line for line in error_lines if line and "Your code has been rated" not in line]
        
        return {
            "issues": issues,
            "errors": errors,
            "return_code": result.returncode,
            "command": " ".join(cmd)
        }
        
    except subprocess.TimeoutExpired:
        raise AnalysisError(f"Pylint timed out after 30 seconds for {filepath}")
    except Exception as e:
        raise AnalysisError(f"Failed to run pylint on {filepath}: {str(e)}")


def _calculate_pylint_score(issues: List[Dict]) -> Tuple[float, Dict[str, int]]:
    """
    Calculate Pylint score from issues.
    Pylint scoring: 10 - (total_errors * 0.1) - (total_warnings * 0.01) - ...
    
    Simplified scoring based on common practice.
    """
    if not issues:
        return 10.0, {"total": 0}
    
    # Count issues by type
    counts = {
        "error": 0,
        "warning": 0,
        "convention": 0,
        "refactor": 0,
        "fatal": 0,
        "info": 0,
        "total": len(issues)
    }
    
    for issue in issues:
        severity = issue.get("type", "").lower()
        if severity in counts:
            counts[severity] += 1
    
    # Base score reduced by issue severity
    base_score = 10.0
    deductions = (
        counts["fatal"] * 2.0 +    # Fatal issues are critical
        counts["error"] * 1.0 +     # Errors are serious
        counts["warning"] * 0.5 +   # Warnings are moderate
        counts["convention"] * 0.1 +  # Convention issues are minor
        counts["refactor"] * 0.2 +  # Refactoring suggestions
        counts["info"] * 0.05       # Info is trivial
    )
    
    # Cap score between 0 and 10
    final_score = max(0.0, min(10.0, base_score - deductions))
    
    return round(final_score, 2), counts


def run_pylint_analysis(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Run Pylint analysis on a Python file and return comprehensive results.
    
    Args:
        filepath: Path to Python file
        
    Returns:
        Dictionary with analysis results:
        {
            "score": 7.5,
            "issues": [...],
            "issue_counts": {...},
            "file_info": {...},
            "recommendations": [...]
        }
        
    Raises:
        SecurityError: If file is outside sandbox
        AnalysisError: If analysis fails
    """
    start_time = datetime.now()
    
    try:
        # Validate path and get absolute path
        abs_path = validate_sandbox_path(filepath)
        
        # Check file exists and is Python
        if not abs_path.exists():
            raise AnalysisError(f"File does not exist: {abs_path}")
        
        if not abs_path.suffix == '.py':
            logger.warning(f"File {abs_path} is not a .py file, analyzing anyway")
        
        # Read file for context
        file_content = safe_read_file(abs_path)
        
        # Run pylint
        pylint_result = _run_pylint_command(abs_path)
        issues = pylint_result["issues"]
        
        # Calculate score
        score, issue_counts = _calculate_pylint_score(issues)
        
        # Generate recommendations
        recommendations = _generate_recommendations(issues, score)
        
        # Prepare results
        result = {
            "score": score,
            "issues": issues[:20],  # Limit issues for logging
            "issue_counts": issue_counts,
            "file_info": {
                "filename": abs_path.name,
                "file_size": len(file_content),
                "lines": file_content.count('\n') + 1,
            },
            "recommendations": recommendations[:5],  # Top 5 recommendations
            "metadata": {
                "analysis_time": (datetime.now() - start_time).total_seconds(),
                "pylint_errors": pylint_result["errors"],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Log the analysis
        _log_analysis_operation(
            filepath=abs_path,
            score=score,
            issue_counts=issue_counts,
            success=True,
            duration=(datetime.now() - start_time).total_seconds()
        )
        
        return result
        
    except (SecurityError, FileOpsError) as e:
        _log_analysis_operation(
            filepath=Path(filepath),
            score=0.0,
            issue_counts={"error": 1, "total": 1},
            success=False,
            error=str(e)
        )
        raise
    except Exception as e:
        _log_analysis_operation(
            filepath=Path(filepath),
            score=0.0,
            issue_counts={"error": 1, "total": 1},
            success=False,
            error=f"Unexpected error: {str(e)}"
        )
        raise AnalysisError(f"Analysis failed for {filepath}: {str(e)}")


def analyze_directory(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze all Python files in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Summary of directory analysis
    """
    from .file_ops import safe_list_files
    
    try:
        abs_dir = validate_sandbox_path(directory)
        
        # List Python files
        py_files = safe_list_files(abs_dir, pattern="*.py")
        
        if not py_files:
            return {
                "directory": str(abs_dir),
                "total_files": 0,
                "average_score": 0.0,
                "file_analyses": []
            }
        
        # Analyze each file
        file_analyses = []
        total_score = 0.0
        
        for py_file in py_files[:10]:  # Limit to 10 files for performance
            try:
                analysis = run_pylint_analysis(py_file)
                file_analyses.append({
                    "file": py_file.name,
                    "score": analysis["score"],
                    "issues": analysis["issue_counts"]["total"]
                })
                total_score += analysis["score"]
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
                file_analyses.append({
                    "file": py_file.name,
                    "score": 0.0,
                    "issues": 0,
                    "error": str(e)
                })
        
        # Calculate average
        avg_score = total_score / len(file_analyses) if file_analyses else 0.0
        
        return {
            "directory": str(abs_dir),
            "total_files": len(py_files),
            "analyzed_files": len(file_analyses),
            "average_score": round(avg_score, 2),
            "file_analyses": file_analyses
        }
        
    except Exception as e:
        raise AnalysisError(f"Directory analysis failed for {directory}: {str(e)}")


def _generate_recommendations(issues: List[Dict], score: float) -> List[str]:
    """
    Generate human-readable recommendations from pylint issues.
    """
    if not issues:
        return ["Code quality is excellent! No issues found."]
    
    recommendations = []
    
    # Group issues by type for better recommendations
    error_issues = [i for i in issues if i.get("type", "").lower() == "error"]
    warning_issues = [i for i in issues if i.get("type", "").lower() == "warning"]
    
    # Score-based recommendations
    if score < 5.0:
        recommendations.append("Critical: Code needs major refactoring. Focus on fixing errors first.")
    elif score < 7.0:
        recommendations.append("Needs improvement: Address major warnings and conventions.")
    elif score < 9.0:
        recommendations.append("Good: Fix minor conventions for perfect score.")

    # Issue-type specific recommendations
    if error_issues:
        top_error = error_issues[0]
        recommendations.append(f"Critical: Fix error {top_error.get('message', 'Unknown error')} (line {top_error.get('line', '?')})")
    if warning_issues:
        top_warning = warning_issues[0]
        recommendations.append(f"Address warning: {top_warning.get('message', 'Unknown warning')}")
    
    # Common issue patterns
    for issue in issues[:3]:  # Top 3 issues
        symbol = issue.get("symbol", "")
        if symbol == "missing-docstring":
            recommendations.append("Add docstrings to functions and classes")
            break
        elif symbol == "line-too-long":
            recommendations.append("Break long lines (max 79 characters)")
            break
        elif symbol == "unused-import":
            recommendations.append("Remove unused imports")
            break
    
    return recommendations[:5]  # Return top 5


def _log_analysis_operation(
    filepath: Path,
    score: float,
    issue_counts: Dict[str, int],
    success: bool,
    duration: float = 0.0,
    error: str = None
) -> None:
    """
    Log analysis operation using project logger.
    """
    try:
        status = "SUCCESS" if success else "FAILURE"
        
        details = {
            "file": str(filepath),
            "score": score,
            "issue_counts": issue_counts,
            "duration_seconds": round(duration, 3),
            "input_prompt": f"Analyze code quality of {filepath.name}",  # Required
            "output_response": f"Score: {score}/10, Issues: {issue_counts.get('total', 0)}"  # Required
        }
        
        if error:
            details["error"] = error
        
        log_experiment(
            agent_name="Toolsmith",
            model_used="pylint",  # Not an LLM, but we need to specify
            action=ActionType.ANALYSIS,
            details=details,
            status=status
        )
        
        # Also log to console
        if success:
            logger.info(f"Analysis complete: {filepath.name} -> Score: {score}/10")
        else:
            logger.error(f"Analysis failed: {filepath.name} -> {error}")
            
    except Exception as e:
        logger.error(f"Failed to log analysis: {e}")


def compare_code_quality(file1: Union[str, Path], file2: Union[str, Path]) -> Dict[str, Any]:
    """
    Compare quality between two code versions.
    Useful for measuring improvement after fixes.
    
    Args:
        file1: Original file
        file2: Fixed file
        
    Returns:
        Comparison results
    """
    try:
        analysis1 = run_pylint_analysis(file1)
        analysis2 = run_pylint_analysis(file2)
        
        improvement = analysis2["score"] - analysis1["score"]
        
        return {
            "original_score": analysis1["score"],
            "new_score": analysis2["score"],
            "improvement": round(improvement, 2),
            "improvement_percent": round((improvement / max(analysis1["score"], 0.1)) * 100, 1),
            "issues_reduced": analysis1["issue_counts"]["total"] - analysis2["issue_counts"]["total"],
            "original_issues": analysis1["issue_counts"],
            "new_issues": analysis2["issue_counts"]
        }
        
    except Exception as e:
        raise AnalysisError(f"Comparison failed: {str(e)}")