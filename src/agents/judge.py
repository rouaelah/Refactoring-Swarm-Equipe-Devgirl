"""
Judge Agent - Executes tests and validates fixes.
Uses tools from src.tools.testing
"""
import time
from typing import Dict, Any, Tuple

from src.utils.logger import log_experiment, ActionType
from src.tools.testing import run_pytest_on_file, run_tests_in_directory
from src.tools.file_ops import safe_read_file

class JudgeAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def run_tests(self, target: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run tests on target (file or directory).
        Returns (success, test_results)
        """
        try:
            print(f"  🧪 Running tests on {target}...")
            start_time = time.time()
            
            # Determine if target is file or directory
            import os
            if os.path.isfile(target):
                test_results = run_pytest_on_file(target, timeout=30)
            else:
                test_results = run_tests_in_directory(target, timeout=30)
            
            elapsed = time.time() - start_time
            
            # Determine success
            success = test_results.get("success", False)
            
            # Log the test execution
            log_experiment(
                agent_name="Judge",
                model_used="pytest",  # Testing tool, not LLM
                action=ActionType.DEBUG if not success else ActionType.GENERATION,
                details={
                    "test_target": target,
                    "tests_passed": success,
                    "test_summary": test_results.get("summary", "No summary"),
                    "total_tests": test_results.get("total_tests", 0),
                    "passed_tests": test_results.get("passed", 0),
                    "failed_tests": test_results.get("failed", 0),
                    "execution_time": elapsed,
                    "input_prompt": f"Run tests on {target}",
                    "output_response": test_results.get("summary", "Test execution complete")
                },
                status="SUCCESS" if success else "FAILURE"
            )
            
            if success:
                print(f"    ✅ Tests passed: {test_results.get('summary', 'All tests passed')}")
            else:
                print(f"    ❌ Tests failed: {test_results.get('summary', 'Some tests failed')}")
            
            return success, test_results
            
        except Exception as e:
            print(f"    ❌ Test execution error: {e}")
            
            # Log the failure
            log_experiment(
                agent_name="Judge",
                model_used="pytest",
                action=ActionType.DEBUG,
                details={
                    "test_target": target,
                    "error": str(e),
                    "input_prompt": f"Run tests on {target}",
                    "output_response": f"Error: {e}"
                },
                status="FAILURE"
            )
            
            return False, {"error": str(e), "success": False}
    
    def generate_test_feedback(self, test_results: Dict[str, Any]) -> str:
        """Generate human-readable feedback from test results"""
        if test_results.get("success", False):
            return "All tests passed successfully!"
        
        # Extract error information
        error_msg = test_results.get("error", "")
        summary = test_results.get("summary", "")
        
        # Try to extract specific test failures
        failed_tests = []
        for detail in test_results.get("details", []):
            if isinstance(detail, dict):
                if detail.get("outcome", "").lower() == "failed":
                    failed_tests.append(detail.get("name", "Unknown test"))
        
        feedback = f"Test failures detected:\n"
        feedback += f"- Summary: {summary}\n"
        
        if error_msg:
            feedback += f"- Error: {error_msg[:200]}\n"
        
        if failed_tests:
            feedback += f"- Failed tests: {', '.join(failed_tests[:5])}\n"
            if len(failed_tests) > 5:
                feedback += f"  ... and {len(failed_tests) - 5} more\n"
        
        return feedback
    
    def validate_code_quality(self, file_path: str, original_score: float) -> Tuple[bool, float]:
        """
        Validate that code quality improved.
        Returns (improved, new_score)
        """
        try:
            # We would need to re-run pylint here
            # For now, this is a placeholder
            print(f"    📊 Validating quality improvement for {file_path.split('/')[-1]}...")
            
            # In a real implementation, you would:
            # 1. Re-run pylint analysis
            # 2. Compare with original_score
            # 3. Return True if score improved
            
            # For now, assume improvement if file was modified
            from src.tools.analysis import run_pylint_analysis
            new_analysis = run_pylint_analysis(file_path)
            new_score = new_analysis.get("score", 0.0)
            
            improved = new_score > original_score
            
            if improved:
                print(f"      ✅ Quality improved: {original_score:.1f} → {new_score:.1f}")
            else:
                print(f"      ⚠️  Quality didn't improve: {original_score:.1f} → {new_score:.1f}")
            
            return improved, new_score
            
        except Exception as e:
            print(f"      ❌ Quality validation failed: {e}")
            return False, original_score