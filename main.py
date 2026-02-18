#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Chargement de l'environnement
load_dotenv()

# Vérification impérative de la clé
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERREUR: GOOGLE_API_KEY manquante dans le fichier .env")
    sys.exit(1)

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.auditor import AuditorAgent
from src.agents.fixer import FixerAgent
from src.agents.judge import JudgeAgent
from src.tools.llm_client import get_llm_client

def main():
    parser = argparse.ArgumentParser(description="Refactoring Swarm - Orchestrator")
    
    parser.add_argument("--target_dir", required=True, help="Dossier contenant le code à refactoriser")
    parser.add_argument("--max_iterations", type=int, default=10, help="Limite d'auto-guérison (max 10)")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Modèle Gemini officiel")
    parser.add_argument("--verbose", action="store_true", help="Mode détaillé")

    args = parser.parse_args()
    target_path = Path(args.target_dir).resolve()

    # --- INITIALISATION ---
    print(f"🤖 Initialisation du système avec le modèle: {args.model}")
    try:
        # On passe verbose pour voir les logs si besoin
        llm_client = get_llm_client(model_name=args.model, temperature=0.1)
    except Exception as e:
        print(f"❌ Erreur client LLM: {e}")
        sys.exit(1)

    auditor = AuditorAgent(llm_client)
    fixer = FixerAgent(llm_client)
    judge = JudgeAgent(llm_client)

    # --- ETAPE 1: AUDIT ---
    print(f"\n🔍 ÉTAPE 1: ANALYSE INITIALE DU CODE DANS {target_path}")
    analysis_results = auditor.analyze_codebase(str(target_path))
    
    if not analysis_results:
        print("⚠️ Aucun fichier Python détecté ou erreur d'analyse.")
        sys.exit(1)

    # --- ETAPE 2: BOUCLE DE SELF-HEALING ---
    print("\n🔧 ÉTAPE 2: BOUCLE DE CORRECTION ET TESTS")
    iteration = 1
    success = False
    feedback_erreurs = None

    

    while iteration <= args.max_iterations and not success:
        print(f"\n--- 🔄 Itération {iteration}/{args.max_iterations} ---")
        
        # A. Correction de chaque fichier audité
        for plan in analysis_results:
            # Extraction sécurisée du chemin
            file_path = plan.get("file_path")
            if not file_path:
                continue
                
            print(f"🛠️ Fixer: Travail sur {file_path}...")
            fixer.fix_file(file_path, str(plan), feedback_erreurs)

        # B. Test et Jugement global du dossier
        print("🧪 Judge: Lancement de la suite de tests...")
        success, test_results = judge.run_tests(str(target_path))

        if success:
            print("✅ SUCCÈS: Tous les tests passent !")
            # Validation bonus de l'amélioration
            for plan in analysis_results:
                f_path = plan.get("file_path")
                old_s = plan.get("pylint_score", 0.0)
                if f_path:
                    judge.validate_improvement(f_path, old_s)
            break
        else:
            print("❌ ÉCHEC: Des tests ne passent pas encore.")
            feedback_erreurs = judge.generate_failure_report(test_results)
            iteration += 1
            if iteration <= args.max_iterations:
                print(f"🔁 Nouvelle tentative de correction basée sur les erreurs de test...")

    # --- ETAPE 3: RAPPORT FINAL ---
    print("\n" + "="*50)
    if success:
        print("🎉 MISSION RÉUSSIE : Code propre et validé.")
    else:
        print("🛑 ÉCHEC : Limite d'itérations atteinte sans succès complet.")
    
    print(f"📊 Logs disponibles dans: logs/experiment_data.json")
    print("="*50)

if __name__ == "__main__":
    main()