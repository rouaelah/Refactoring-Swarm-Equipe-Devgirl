# test_data.py 
import json
import os
from datetime import datetime
import uuid

TEST_LOG_FILE = "logs/experiment_data.json"

def create_test_data():
    os.makedirs("logs", exist_ok=True)
    
    test_data = [
        {
            "id": str(uuid.uuid4()),
            "agent": "AuditorAgent",
            "model": "gemini-2.0-flash",
            "action": "CODE_ANALYSIS",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "file_analyzed": "buggy_code.py",
                "input_prompt": "Analyse ce code Python et trouve les bugs...",
                "output_response": "J'ai trouvé 3 problèmes : 1. Pas de gestion d'erreur...",
                "issues_found": 3,
                "pylint_score_before": 5.2
            },
            "status": "SUCCESS"
        },
        {
            "id": str(uuid.uuid4()),
            "agent": "FixerAgent", 
            "model": "gemini-2.0-flash",
            "action": "FIX",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "file_modified": "buggy_code.py",
                "input_prompt": "Corrige les problèmes suivants...",
                "output_response": "J'ai ajouté un try-except pour la division par zéro...",
                "changes_made": ["Added try-except", "Added docstring"]
            },
            "status": "SUCCESS"
        },
        {
            "id": str(uuid.uuid4()),
            "agent": "JudgeAgent",
            "model": "gemini-2.0-flash",
            "action": "DEBUG",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "test_file": "test_buggy_code.py",
                "input_prompt": "Pourquoi ce test échoue-t-il ?",
                "output_response": "Le test échoue car la fonction retourne None quand...",
                "test_status": "FAILED",
                "error_message": "AssertionError: Expected 10, got None"
            },
            "status": "SUCCESS"
        }
    ]

    with open(TEST_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("Jeu de données de test créé avec succès.")
    print(f"   {len(test_data)} entrées ajoutées.")
    print("   Utilise 'python quality/verify.py' pour valider.")


if __name__ == "__main__":
    create_test_data()