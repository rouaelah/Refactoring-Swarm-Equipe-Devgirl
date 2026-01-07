"""Integration test for analysis tools"""
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

from src.tools.analysis import (
    run_pylint_analysis,
    analyze_directory,
    compare_code_quality,
    _calculate_pylint_score,
    _generate_recommendations,
)
from src.tools.file_ops import safe_write_file
from src.tools.security import get_sandbox_root

def create_test_files():
    """Create test Python files"""
    sandbox = get_sandbox_root()
    test_dir = sandbox / "analysis_integration_test"
    test_dir.mkdir(exist_ok=True)
    
    # Clean previous test
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # File 1: Good Python code
    good_code = '''"""
A well-written Python module for demonstration.
"""

import math
from typing import List, Optional


def calculate_statistics(numbers: List[float]) -> dict:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of float numbers
        
    Returns:
        Dictionary with mean, median, and standard deviation
    """
    if not numbers:
        return {"mean": 0, "median": 0, "std_dev": 0}
    
    # Calculate mean
    mean = sum(numbers) / len(numbers)
    
    # Calculate median
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = math.sqrt(variance)
    
    return {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std_dev": round(std_dev, 2)
    }


class StatisticsCalculator:
    """Class for statistical calculations."""
    
    def __init__(self, data: Optional[List[float]] = None):
        """Initialize with optional data."""
        self.data = data or []
    
    def add_number(self, number: float) -> None:
        """Add a number to the dataset."""
        self.data.append(number)
    
    def compute_all(self) -> dict:
        """Compute all statistics."""
        return calculate_statistics(self.data)


if __name__ == "__main__":
    # Example usage
    calculator = StatisticsCalculator([1, 2, 3, 4, 5])
    print(calculator.compute_all())
'''
    
    good_file = test_dir / "good_example.py"
    safe_write_file(good_file, good_code)
    
    # File 2: Bad Python code
    bad_code = '''import os,sys,math,json,re # bad imports
x=1
y=2
z=x+y

class bad_class:
    def __init__(self):
        self.attr=123
    
    def method1(self,a,b):
        result=a+b
        return result

def bad_func():
    unused=5
    print("hello")
    return

# Missing type hints
# Poor formatting
# No docstrings
'''
    
    bad_file = test_dir / "bad_example.py"
    safe_write_file(bad_file, bad_code)
    
    # File 3: Fixed version of bad code
    fixed_code = '''"""
Improved version of the code with better practices.
"""

import math
from typing import Tuple


class GoodClass:
    """A well-named class with proper documentation."""
    
    def __init__(self):
        """Initialize the class."""
        self.attribute = 123
    
    def method(self, a: int, b: int) -> int:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        result = a + b
        return result


def good_function() -> str:
    """
    A good function example.
    
    Returns:
        Greeting message
    """
    message = "Hello, World!"
    return message


if __name__ == "__main__":
    obj = GoodClass()
    print(obj.method(10, 20))
    print(good_function())
'''
    
    fixed_file = test_dir / "fixed_example.py"
    safe_write_file(fixed_file, fixed_code)
    
    return test_dir, good_file, bad_file, fixed_file


def run_integration_test():
    """Run full integration test"""
    print("Analysis Tools Integration Test")
    print("=" * 60)
    
    try:
        # Create test files
        test_dir, good_file, bad_file, fixed_file = create_test_files()
        
        print(f"\nTest directory: {test_dir}")
        
        # Test 1: Analyze good code
        print("\n1 Analyzing GOOD Python code...")
        try:
            good_result = run_pylint_analysis(good_file)
            print(f"   Score: {good_result['score']}/10")
            print(f"   Issues found: {good_result['issue_counts']['total']}")
            
            # Show recommendations
            if good_result['recommendations']:
                print("   Recommendations:")
                for rec in good_result['recommendations'][:3]:
                    print(f"     - {rec}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 2: Analyze bad code
        print("\n2 Analyzing BAD Python code...")
        try:
            bad_result = run_pylint_analysis(bad_file)
            print(f"   Score: {bad_result['score']}/10")
            print(f"   Issues found: {bad_result['issue_counts']['total']}")
            
            if bad_result['issue_counts']['total'] > 0:
                print(f"   Issue breakdown:")
                for issue_type, count in bad_result['issue_counts'].items():
                    if count > 0 and issue_type != 'total':
                        print(f"     - {issue_type}: {count}")
            
            # Show top issues
            if bad_result['issues']:
                print(f"   Top issue: {bad_result['issues'][0].get('message', 'Unknown')}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 3: Compare code quality
        print("\n3 Comparing code quality (before/after fix)...")
        try:
            comparison = compare_code_quality(bad_file, fixed_file)
            print(f"   Original score: {comparison['original_score']}/10")
            print(f"   New score: {comparison['new_score']}/10")
            print(f"   Improvement: +{comparison['improvement']}")
            print(f"   Issues reduced: {comparison['issues_reduced']}")
            
            if comparison['improvement'] > 0:
                print("   Quality improved!")
            else:
                print("   No improvement detected")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 4: Analyze directory
        print("\n4 Analyzing directory (all Python files)...")
        try:
            dir_result = analyze_directory(test_dir)
            print(f"   Files found: {dir_result['total_files']}")
            print(f"   Files analyzed: {dir_result['analyzed_files']}")
            print(f"   Average score: {dir_result['average_score']}/10")
            
            if dir_result['file_analyses']:
                print("   Individual file scores:")
                for file_analysis in dir_result['file_analyses']:
                    print(f"     - {file_analysis['file']}: {file_analysis['score']}/10")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 5: Score calculation logic
        print("\n5 Testing score calculation logic...")
        try:
            # Mock issues
            test_issues = [
                {"type": "error", "message": "Test error"},
                {"type": "warning", "message": "Test warning"},
            ]
            score, counts = _calculate_pylint_score(test_issues)
            print(f"   Calculated score: {score}/10")
            print(f"   Issue counts: {counts}")
            assert score < 10.0, "Score should be reduced with issues"
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test 6: Recommendation generation
        print("\n6 Testing recommendation generation...")
        try:
            test_issues = [
                {"type": "error", "symbol": "syntax-error", "message": "Syntax error", "line": 1},
                {"type": "warning", "symbol": "line-too-long", "message": "Line too long", "line": 50},
            ]
            recs = _generate_recommendations(test_issues, 6.5)
            print(f"   Generated {len(recs)} recommendations:")
            for i, rec in enumerate(recs[:3], 1):
                print(f"     {i}. {rec}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        print("\n" + "=" * 60)
        print("All analysis integration tests completed!")
        
        # Show log count
        log_file = Path("logs") / "experiment_data.json"
        if log_file.exists():
            try:
                import json
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        analysis_entries = [e for e in data if e.get('action') == 'CODE_ANALYSIS']
                        print(f"\nAnalysis log entries created: {len(analysis_entries)}")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)