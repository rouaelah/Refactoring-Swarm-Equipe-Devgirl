"""
Fixer Agent - Applies refactoring plans to fix code.
Uses tools from src.tools.file_ops and src.tools.security
"""
from typing import Optional, Dict, Any, List

from src.utils.logger import log_experiment, ActionType
from src.tools.file_ops import safe_read_file, safe_write_file
from src.tools.security import validate_sandbox_path, is_within_sandbox

class FixerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def fix_file(self, file_path: str, refactoring_plan: str, 
                 previous_errors: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply refactoring to a single file.
        Returns dict with results.
        """
        try:
            # Security check
            if not is_within_sandbox(file_path):
                raise PermissionError(f"Cannot write outside sandbox: {file_path}")
            
            # Read original code
            original_code = safe_read_file(file_path)
            
            # Create LLM prompt
            prompt = self._create_fix_prompt(original_code, refactoring_plan, previous_errors)
            
            # Get fixed code from LLM
            fixed_code = self.llm_client.generate(prompt)
            
            # Clean up the response (remove markdown if present)
            fixed_code = self._clean_llm_response(fixed_code)
            
            # Write fixed code
            safe_write_file(file_path, fixed_code)
            
            # Verify the fix was applied
            written_code = safe_read_file(file_path)
            
            result = {
                "file_path": file_path,
                "original_length": len(original_code),
                "fixed_length": len(fixed_code),
                "fix_applied": fixed_code.strip() != original_code.strip(),
                "verification_passed": written_code.strip() == fixed_code.strip()
            }
            
            # Log the fix
            log_experiment(
                agent_name="Fixer",
                model_used="gemini-2.0-flash",
                action=ActionType.FIX,
                details={
                    "file": file_path,
                    "fix_applied": result["fix_applied"],
                    "verification_passed": result["verification_passed"],
                    "input_prompt": prompt[:500],  # Truncate for logs
                    "output_response": fixed_code[:500]  # Truncate
                },
                status="SUCCESS"
            )
            
            print(f"    🔧 Fixed {file_path.split('/')[-1]} - Changes: {result['fix_applied']}")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed to fix {file_path}: {e}")
            
            # Log the failure
            log_experiment(
                agent_name="Fixer",
                model_used="gemini-2.0-flash",
                action=ActionType.FIX,
                details={
                    "file": file_path,
                    "error": str(e),
                    "input_prompt": "Fix code",
                    "output_response": f"Error: {e}"
                },
                status="FAILURE"
            )
            
            raise
    
    def _clean_llm_response(self, code: str) -> str:
        """Remove markdown code blocks if present"""
        # Remove ```python and ``` markers
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]  # Remove ```python
        elif code.startswith("```"):
            code = code[3:]  # Remove ```
        
        if code.endswith("```"):
            code = code[:-3]  # Remove trailing ```
        
        return code.strip()
    
    def _create_fix_prompt(self, code: str, plan: str, previous_errors: Optional[str]) -> str:
        """Create prompt for fixing code based on plan"""
        
        base_prompt = f"""You are a Python expert. Fix this code according to the refactoring plan.

ORIGINAL CODE:
{code}

REFACTORING PLAN:
{plan}
"""
        
        if previous_errors:
            base_prompt += f"""

PREVIOUS TEST ERRORS (fix these):
{previous_errors}
"""
        
        base_prompt += """

INSTRUCTIONS:
1. Apply ALL fixes from the refactoring plan
2. Keep the same functionality
3. Fix any syntax errors
4. Add docstrings if missing
5. Follow PEP 8 guidelines

Return ONLY the fixed Python code, no explanations, no markdown code blocks."""

        return base_prompt
    
    def batch_fix_files(self, files_to_fix: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fix multiple files.
        files_to_fix: List of dicts with 'file_path' and 'refactoring_plan'
        """
        results = {
            "total_files": len(files_to_fix),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for file_info in files_to_fix:
            try:
                result = self.fix_file(
                    file_info["file_path"],
                    file_info["refactoring_plan"],
                    file_info.get("previous_errors")
                )
                results["details"].append(result)
                if result["fix_applied"]:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
            except Exception:
                results["failed"] += 1
        
        return results