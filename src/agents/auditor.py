"""
Auditor Agent - Analyzes code quality and creates refactoring plans.
Uses tools from src.tools.analysis
"""
import os
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import log_experiment, ActionType
from src.tools.analysis import run_pylint_analysis, analyze_directory
from src.tools.file_ops import safe_read_file, safe_list_files
from src.tools.security import validate_sandbox_path

class AuditorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def analyze_codebase(self, target_dir: str) -> List[Dict[str, Any]]:
        """
        Analyze all Python files in target directory.
        Returns list of analysis results for each file.
        """
        try:
            # Validate directory is in sandbox
            abs_target = validate_sandbox_path(target_dir)
            
            # Get all Python files
            python_files = safe_list_files(abs_target, pattern="*.py", recursive=True)
            
            if not python_files:
                print(f"⚠️  No Python files found in {target_dir}")
                return []
            
            print(f"🔍 Found {len(python_files)} Python files to analyze")
            
            analysis_results = []
            
            for file_path in python_files:
                try:
                    print(f"  📄 Analyzing {file_path.name}...")
                    
                    # Read file content
                    file_content = safe_read_file(file_path)
                    
                    # Run Pylint analysis using the provided tool
                    pylint_result = run_pylint_analysis(file_path)
                    
                    # Prepare LLM prompt based on analysis
                    prompt = self._create_analysis_prompt(file_content, pylint_result)
                    
                    # Get refactoring plan from LLM
                    refactoring_plan = self.llm_client.generate(prompt)
                    
                    # Store result
                    result = {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "original_code": file_content,
                        "pylint_analysis": pylint_result,
                        "refactoring_plan": refactoring_plan,
                        "score": pylint_result.get("score", 0.0),
                        "issues_count": pylint_result.get("issue_counts", {}).get("total", 0)
                    }
                    
                    analysis_results.append(result)
                    
                    # Log the analysis
                    log_experiment(
                        agent_name="Auditor",
                        model_used="gemini-2.0-flash",
                        action=ActionType.ANALYSIS,
                        details={
                            "file_analyzed": str(file_path),
                            "pylint_score": pylint_result.get("score", 0.0),
                            "issues_found": pylint_result.get("issue_counts", {}).get("total", 0),
                            "input_prompt": prompt[:500],  # Limit size for logs
                            "output_response": refactoring_plan[:500]  # Limit size
                        },
                        status="SUCCESS"
                    )
                    
                    print(f"    ✅ Score: {pylint_result.get('score', 0.0)}/10 - Issues: {pylint_result.get('issue_counts', {}).get('total', 0)}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to analyze {file_path.name}: {e}")
                    # Log the failure
                    log_experiment(
                        agent_name="Auditor",
                        model_used="gemini-2.0-flash",
                        action=ActionType.ANALYSIS,
                        details={
                            "file_analyzed": str(file_path),
                            "error": str(e),
                            "input_prompt": "File analysis",
                            "output_response": f"Error: {e}"
                        },
                        status="FAILURE"
                    )
            
            # Sort by worst score first
            analysis_results.sort(key=lambda x: x["score"])
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Auditor failed: {e}")
            raise
    
    def _create_analysis_prompt(self, code: str, pylint_result: Dict[str, Any]) -> str:
        """Create prompt for LLM based on code analysis"""
        
        # Extract recommendations from pylint
        recommendations = "\n".join(pylint_result.get("recommendations", []))
        
        # Extract key issues
        issues = []
        for issue in pylint_result.get("issues", [])[:10]:  # Limit to 10 issues
            issues.append(f"- Line {issue.get('line', '?')}: {issue.get('message', 'Unknown')}")
        
        issues_text = "\n".join(issues) if issues else "No specific issues found."
        
        prompt = f"""You are a senior Python code reviewer. Analyze this code and provide a detailed refactoring plan.

CODE TO REVIEW:  
{code}

PYLINT ANALYSIS RESULTS:
- Score: {pylint_result.get('score', 0.0)}/10
- Total issues: {pylint_result.get('issue_counts', {}).get('total', 0)}
- Key issues:
{issues_text}

RECOMMENDATIONS:
{recommendations}

Please provide a STRUCTURED refactoring plan with:
1. CRITICAL BUGS: List any bugs that must be fixed immediately
2. CODE SMELLS: Identify poor patterns or design issues
3. PERFORMANCE: Suggest performance improvements if needed
4. READABILITY: Suggestions for better naming, comments, structure
5. TESTING: What tests should be added or modified

Return only the refactoring plan, no explanations or markdown formatting."""
        
        return prompt
    
    def get_summary_report(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate summary report of analysis"""
        if not analysis_results:
            return "No files analyzed."
        
        total_files = len(analysis_results)
        avg_score = sum(r["score"] for r in analysis_results) / total_files
        total_issues = sum(r["issues_count"] for r in analysis_results)
        
        worst_files = sorted(analysis_results, key=lambda x: x["score"])[:3]
        
        report = f"""
📊 AUDIT SUMMARY
================
Files analyzed: {total_files}
Average Pylint score: {avg_score:.2f}/10
Total issues found: {total_issues}

TOP 3 FILES NEEDING ATTENTION:
"""
        for i, file in enumerate(worst_files, 1):
            report += f"{i}. {file['file_name']} - Score: {file['score']:.1f}/10 - Issues: {file['issues_count']}\n"
        
        return report