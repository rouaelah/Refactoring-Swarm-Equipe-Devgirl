import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.utils.logger import log_experiment, ActionType
from src.tools.file_ops import safe_read_file, safe_write_file
from src.tools.security import validate_sandbox_path, is_within_sandbox

class FixerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model_name = "gemini-1.5-flash"
        # Chargement des instructions système de l'Ingénieur Prompt
        self.system_instructions = self._load_system_prompts()

    def _load_system_prompts(self) -> str:
        """Charge les instructions partagées et spécifiques au correcter."""
        try:
            # On n'utilise PLUS safe_read_file ici pour éviter la SECURITY VIOLATION
            # On utilise le open() standard de Python
            with open("src/prompts/shared_instruction.md", "r", encoding="utf-8") as f:
                shared = f.read()
                
            with open("src/prompts/correcteur.md", "r", encoding="utf-8") as f:
                specific = f.read()
                
            return f"{shared}\n\n{specific}"
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des prompts du Fixer : {e}")
            return "Tu es un expert en refactoring. Réponds uniquement avec le code Python corrigé."

    def fix_file(self, file_path: str, refactoring_plan: str, 
                 previous_errors: Optional[str] = None) -> Dict[str, Any]:
        try:
            abs_path = validate_sandbox_path(file_path)
            original_code = safe_read_file(abs_path)
            
            user_prompt = f"PLAN: {refactoring_plan}\nCODE: {original_code}"
            if previous_errors:
                user_prompt += f"\nERRORS: {previous_errors}"

            # CORRECTION ICI: prompt=
            fixed_code_raw = self.llm_client.generate(
                system_instruction=self.system_instructions,
                prompt=user_prompt
            )

            fixed_code = self._clean_llm_response(fixed_code_raw)
            safe_write_file(abs_path, fixed_code)

            # CORRECTION LOGGER
            log_experiment(
                agent_name="Fixer_Agent",
                model_used=self.model_name,
                action=ActionType.FIX,
                status="SUCCESS",
                details={
                    "input_prompt": user_prompt,
                    "output_response": fixed_code_raw,
                    "file": str(file_path)
                }
            )

            return {"file_path": file_path, "fix_applied": True}
        except Exception as e:
            raise e

        except Exception as e:
            print(f"❌ Erreur Fixer sur {file_path}: {e}")
            log_experiment(
                agent_name="Fixer_Agent",
                model_used=self.model_name,
                action=ActionType.FIX,
                input_prompt="Execution error",
                output_response=str(e),
                status="FAILURE"
            )
            raise

    def _clean_llm_response(self, code: str) -> str:
        """Nettoie les balises ```python du LLM."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def batch_fix_files(self, audit_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Traite une liste de plans d'audit issus de l'Auditeur."""
        results = []
        for plan in audit_plans:
            # On extrait le chemin du fichier depuis le rapport de l'auditeur
            # Note: s'adapte au format JSON défini par ton Prompt Engineer
            file_path = plan.get("audit_report", {}).get("file_analyzed") or plan.get("file_path")
            if file_path:
                res = self.fix_file(file_path, str(plan))
                results.append(res)
        return results