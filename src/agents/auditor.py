import json
import os
from typing import List, Dict, Any
from pathlib import Path

from src.utils.logger import log_experiment, ActionType
from src.tools.analysis import run_pylint_analysis
from src.tools.file_ops import safe_read_file, safe_list_files
from src.tools.security import validate_sandbox_path

class AuditorAgent:
    def __init__(self, llm_client):
        """
        Initialise l'auditeur avec le client LLM (Gemini).
        """
        self.llm_client = llm_client
        self.model_name = "gemini-1.5-flash"  
        # Chargement des instructions système depuis les fichiers Markdown de l'équipe
        self.system_prompt = self._load_prompt_files()

    def _load_prompt_files(self) -> str:
        # 1. On définit les chemins vers tes fichiers de prompts
        shared_path = "src/prompts/shared_instruction.md"
        specific_path = "src/prompts/auditeur.md" # Change selon l'agent
        
        try:
            # 2. Utilise open() standard (SANS passer par safe_read_file)
            with open(shared_path, "r", encoding="utf-8") as f:
                shared = f.read()
            
            with open(specific_path, "r", encoding="utf-8") as f:
                specific = f.read()
                
            return f"{shared}\n\n{specific}"
            
        except FileNotFoundError as e:
            print(f"❌ Erreur : Impossible de trouver le prompt à {e.filename}")
            return "Tu es un expert Python."

    def analyze_codebase(self, target_dir: str) -> List[Dict[str, Any]]:
        analysis_results = []
        abs_target = validate_sandbox_path(target_dir)
        python_files = safe_list_files(abs_target, pattern="*.py", recursive=True)
        
        for file_path in python_files:
            try:
                file_content = safe_read_file(file_path)
                pylint_res = run_pylint_analysis(file_path)
                
                user_message = f"FICHIER: {file_path.name}\nCONTENU:\n{file_content}\nPYLINT: {pylint_res.get('score')}"

                # CORRECTION ICI: prompt= au lieu de user_message=
                raw_response = self.llm_client.generate(
                    system_instruction=self.system_prompt,
                    prompt=user_message 
                )

                # CORRECTION LOGGER: input_prompt est DANS details
                log_experiment(
                    agent_name="Auditor_Agent",
                    model_used=self.model_name,
                    action=ActionType.ANALYSIS,
                    status="SUCCESS",
                    details={
                        "input_prompt": user_message,
                        "output_response": raw_response,
                        "file_path": str(file_path),
                        "pylint_score": pylint_res.get('score')
                    }
                )

                analysis_data = self._parse_json_response(raw_response)
                # On s'assure que le chemin est bien dans l'objet pour le Fixer
                analysis_data["file_path"] = str(file_path)
                analysis_data["score"] = pylint_res.get('score')
                analysis_results.append(analysis_data)
                
            except Exception as e:
                print(f"❌ Erreur audit: {e}")
        return analysis_results

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Nettoie et convertit la réponse du LLM en dictionnaire Python."""
        try:
            # Enlever les balises markdown si présentes
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return {"error": "Format JSON invalide", "raw_text": text}