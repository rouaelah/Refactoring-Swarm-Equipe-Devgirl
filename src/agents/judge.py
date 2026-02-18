import time
from pathlib import Path
from typing import Dict, Any, Tuple

from src.utils.logger import log_experiment, ActionType
from src.tools.testing import run_pytest_on_file, run_tests_in_directory
from src.tools.analysis import run_pylint_analysis
from src.tools.file_ops import safe_read_file

class JudgeAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # Chargement du prompt de testeur sans passer par la sécurité sandbox
        try:
            # On utilise le open() standard de Python car on est dans src/
            with open("src/prompts/testeur.md", "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except Exception as e:
            print(f"⚠️ Impossible de charger le prompt testeur: {e}")
            self.system_prompt = "Tu es un expert QA. Analyse les erreurs de test."

    def run_tests(self, target: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Exécute les tests unitaires et logue le résultat pour le Data Officer.
        """
        print(f"🧪 Judge: Exécution des tests sur {target}...")
        
        # 1. Lancement de Pytest via Toolsmith
        import os
        if os.path.isfile(target):
            test_results = run_pytest_on_file(target, timeout=30)
        else:
            test_results = run_tests_in_directory(target, timeout=30)
        
        success = test_results.get("success", False)

        # 2. LOGGING OBLIGATOIRE (Critère de notation 30%)
        # On utilise ActionType.EVALUATION ou DEBUG selon le résultat
        log_experiment(
            agent_name="Judge_Agent",
            model_used="pytest_engine", 
            action=ActionType.DEBUG if not success else ActionType.EVALUATION,
            status="SUCCESS" if success else "FAILURE",
            details={
                "input_prompt": f"Validation technique de: {target}",
                "output_response": test_results.get("summary", "Pas de résumé"),
                "total_tests": test_results.get("total_tests", 0)
            }
        )
        
        return success, test_results

    def validate_improvement(self, file_path: str, old_score: float) -> Tuple[bool, float]:
        """
        Vérifie si le score Pylint s'est amélioré après correction.
        """
        print(f"📊 Judge: Vérification de l'amélioration du score...")
        analysis = run_pylint_analysis(Path(file_path))
        new_score = analysis.get("score", 0.0)
        
        improved = new_score > old_score
        
        if improved:
            print(f"✅ Amélioration confirmée: {old_score} -> {new_score}")
        else:
            print(f"⚠️ Pas d'amélioration notable du score: {new_score}")
            
        return improved, new_score

    def generate_failure_report(self, test_results: Dict[str, Any]) -> str:
        """
        En cas d'échec, prépare un rapport détaillé pour le Fixer (Feedback Loop).
        """
        summary = test_results.get("summary", "Erreur inconnue")
        details = test_results.get("details", [])
        
        report = f"ECHEC DES TESTS DETECTÉ:\n{summary}\n\nDÉTAILS DES ERREURS:\n"
        
        # Extraction des 3 premières erreurs pour ne pas saturer le prompt
        failures = [d for d in details if isinstance(d, dict) and d.get("outcome") == "failed"]
        for f in failures[:3]:
            report += f"- Test: {f.get('name')}\n  Message: {f.get('message')}\n"
            
        return report