# verify.py:fonction qui verifie les actions des agents  
import json
import os

LOG_FILE = "logs/experiment_data.json"

REQUIRED_FIELDS = ["id", "agent", "model", "action", "timestamp", "details", "status"]

def verify_log_file():
    # 1. Vérifier que le fichier existe
    if not os.path.exists(LOG_FILE):
        print("ERREUR : Le fichier experiment_data.json n'existe pas.")
        return False

    # 2. Charger le fichier JSON
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("ERREUR : Le fichier JSON est mal formé.")
        return False

    # 3. Vérifier que c'est une liste
    if not isinstance(data, list):
        print("ERREUR : Le fichier JSON doit contenir une liste d'actions.")
        return False

    # 4. Vérifier chaque entrée
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"ERREUR : Entrée {i} n'est pas un objet JSON.")
            return False

        # Vérifier tous les champs obligatoires
        for field in REQUIRED_FIELDS:
            if field not in entry:
                print(f"ERREUR : Champ manquant '{field}' dans l'entrée {i}.")
                return False
        
        # VÉRIFICATION SUPPLÉMENTAIRE CRITIQUE : 
        # Vérifier que 'details' contient bien 'input_prompt' et 'output_response'
        # pour certaines actions
        details = entry.get("details", {})
        action = entry.get("action", "")
        
        # Actions qui nécessitent les prompts (selon le logger du prof)
        actions_requiring_prompts = ["CODE_ANALYSIS", "CODE_GEN", "DEBUG", "FIX"]
        
        if action in actions_requiring_prompts:
            if "input_prompt" not in details:
                print(f"ERREUR : Entrée {i} (action {action}) : 'input_prompt' manquant dans 'details'")
                return False
            if "output_response" not in details:
                print(f"ERREUR : Entrée {i} (action {action}) : 'output_response' manquant dans 'details'")
                return False

    print("SUCCÈS : Le fichier de logs est valide selon le schéma imposé.")
    return True