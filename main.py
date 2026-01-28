#!/usr/bin/env python3
"""
Main orchestrator for the Refactoring Swarm.
Coordinates Auditor, Fixer, and Judge agents using LangChain.
"""
import argparse
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis .env
load_dotenv()

# Vérifier que la clé est bien présente
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ GOOGLE_API_KEY manquante dans .env")
    sys.exit(1)


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import agents (using lowercase filenames)
from src.agents.auditor import AuditorAgent
from src.agents.fixer import FixerAgent
from src.agents.judge import JudgeAgent

# Import LangChain Gemini client
try:
    from src.tools.llm_client import get_llm_client
    print("✅ Using LangChain Gemini client")
except ImportError as e:
    print(f"❌ Failed to import LLM client: {e}")
    print("   Make sure langchain-google-genai is installed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Refactoring Swarm - Autonomous Code Refactoring System"
    )
    
    # ARGUMENTS OBLIGATOIRES
    parser.add_argument(
        "--target_dir", 
        required=True,
        help="Directory containing buggy Python code to refactor"
    )
    
    # ARGUMENTS OPTIONNELS
    parser.add_argument(
        "--max_iterations", 
        type=int, 
        default=10,
        help="Maximum self-healing iterations (default: 10)"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
       help="Gemini model to use (default:gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",  # Ceci rend --verbose un flag booléen
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock LLM instead of real API"
    )
    
    # Parse les arguments
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔄 REFACTORING SWARM - Autonomous Code Refactoring")
    print("="*60)
    
    # Validate target directory
    target_path = Path(args.target_dir)
    if not target_path.exists():
        print(f"❌ Target directory not found: {args.target_dir}")
        sys.exit(1)
    
    print(f"🎯 Target: {target_path.resolve()}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"🤖 Model: {args.model}")
    print(f"🔊 Verbose: {args.verbose}")
    print(f"🤖 Mock mode: {args.use_mock}")
    print("-"*60)
    
    # Set mock mode if requested
    if args.use_mock:
        os.environ["USE_MOCK_LLM"] = "true"
        print("⚠️  Using MOCK LLM (no API calls)")
    
    # Initialize LLM client using LangChain
    try:
        llm_client = get_llm_client(model_name=args.model, temperature=0.1)
        print("✅ LLM client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        print("   Check your GOOGLE_API_KEY in .env file")
        sys.exit(1)
    
    # Create agents
    print("🤖 Initializing agents...")
    auditor = AuditorAgent(llm_client)
    fixer = FixerAgent(llm_client)
    judge = JudgeAgent(llm_client)
    print("✅ Agents ready")
    
    # STEP 1: AUDIT
    print("\n" + "="*60)
    print("🔍 STEP 1: CODE AUDIT")
    print("="*60)
    
    try:
        analysis_results = auditor.analyze_codebase(str(target_path))
        
        if not analysis_results:
            print("❌ No Python files found to analyze")
            sys.exit(1)
        
        # Show audit summary
        print(auditor.get_summary_report(analysis_results))
        
    except Exception as e:
        print(f"❌ Audit phase failed: {e}")
        sys.exit(1)
    
    # STEP 2: REFACTORING LOOP
    print("\n" + "="*60)
    print("🔧 STEP 2: REFACTORING & VALIDATION LOOP")
    print("="*60)
    
    iteration = 1
    all_tests_passed = False
    test_errors = None
    
    while iteration <= args.max_iterations and not all_tests_passed:
        print(f"\n🔄 ITERATION {iteration}/{args.max_iterations}")
        
        # Fix files
        print("🔧 Applying fixes...")
        for file_analysis in analysis_results:
            try:
                fix_result = fixer.fix_file(
                    file_analysis["file_path"],
                    file_analysis["refactoring_plan"],
                    test_errors
                )
                
                if args.verbose and fix_result.get("fix_applied"):
                    print(f"   ✅ {file_analysis['file_name']}: {fix_result.get('original_length')} → {fix_result.get('fixed_length')} chars")
                    
            except Exception as e:
                print(f"   ❌ Failed to fix {file_analysis['file_name']}: {e}")
        
        # Test the fixes
        print("🧪 Running tests...")
        tests_passed, test_results = judge.run_tests(str(target_path))
        
        if tests_passed:
            print("\n✅ ALL TESTS PASSED!")
            all_tests_passed = True
            
            # Optional: Validate quality improvement
            if args.verbose:
                print("\n📊 Validating quality improvements...")
                for file_analysis in analysis_results:
                    improved, new_score = judge.validate_code_quality(
                        file_analysis["file_path"],
                        file_analysis["score"]
                    )
            
            break
        else:
            print("\n❌ TESTS FAILED - Preparing for next iteration")
            test_errors = judge.generate_test_feedback(test_results)
            
            if args.verbose:
                print(f"   Error feedback: {test_errors[:200]}...")
            else:
                print(f"   Failed tests: {test_results.get('failed', 0)}/{test_results.get('total_tests', 0)}")
            
            # Update refactoring plans with test errors
            for file_analysis in analysis_results:
                file_analysis["refactoring_plan"] += f"\n\nTEST ERRORS TO FIX:\n{test_errors}"
        
        iteration += 1
    
    # FINAL REPORT
    print("\n" + "="*60)
    print("📊 FINAL REPORT")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 MISSION SUCCESSFUL!")
        print(f"✅ All tests pass after {iteration} iteration(s)")
        print("✅ Code has been refactored and validated")
        exit_code = 0
    else:
        print("⚠️  MISSION PARTIALLY COMPLETE")
        print(f"❌ Could not make all tests pass after {args.max_iterations} iterations")
        print("⚠️  Some improvements may have been made, but tests still fail")
        exit_code = 1
    
    print(f"\n📁 Logs saved to: logs/experiment_data.json")
    print("📊 Check the log file for detailed execution history")
    
    # Force write logs to ensure they're saved
    time.sleep(0.5)  # Small delay for async operations
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()