
import argparse
import os
import sys

# Import futur - pour l'instant on commente pour éviter l'erreur
# from src.auditor_agent import AuditorAgent
# from src.utils.logger import log_experiment, ActionType

def main():
    """Fonction principale - Version SIMPLE pour débuter"""
    
    print("=" * 50)
    print("🤖 REFACTORING SWARM - Système Multi-Agents")
    print("=" * 50)
    
    # 1. Parser les arguments
    parser = argparse.ArgumentParser(
        description="Refactorise du code Python automatiquement"
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        help='Chemin vers le dossier contenant le code à refactoriser'
    )
    
    args = parser.parse_args()
    
    # 2. Validation du dossier
    target_dir = args.target_dir
    print(f"🚀 DEMARRAGE SUR : {target_dir}")
    
    if not os.path.exists(target_dir):
        print(f"❌ ERREUR: Le dossier '{target_dir}' n'existe pas!")
        sys.exit(1)
    
    if not os.path.isdir(target_dir):
        print(f"❌ ERREUR: '{target_dir}' n'est pas un dossier!")
        sys.exit(1)
    
    # 3. Chercher les fichiers Python
    python_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    
    print(f"📂 {len(python_files)} fichiers Python trouvés")
    
    # 4. Afficher la liste des fichiers
    if python_files:
        print("\n📄 Liste des fichiers à analyser:")
        for i, file_path in enumerate(python_files, 1):
            print(f"  {i}. {file_path}")
    
    print("\n" + "=" * 50)
    print("🔍 LANCEMENT DE L'ANALYSE AVEC L'AUDITEUR")
    print("=" * 50)
    
    try:
        # Importer l'Auditeur
        from src.auditeur_agent import AuditorAgent
        
        # Créer l'instance
        auditor = AuditorAgent()
        print("✅ AuditorAgent initialisé")
        
        # Analyser chaque fichier
        for i, file_path in enumerate(python_files, 1):
            print(f"\n[{i}/{len(python_files)}] Analyse: {os.path.basename(file_path)}")
            result = auditor.analyze_single_file(file_path)
            
            if "error" in result:
                print(f"   ❌ Erreur: {result['error']}")
            else:
                print(f"   ✅ {result['summary']}")
                # Afficher un aperçu des problèmes trouvés
                issues = result.get('issues', '').split('\n')
                for issue in issues[:3]:  # Afficher seulement 3 premiers problèmes
                    if issue.strip():
                        print(f"      • {issue.strip()}")
        
        print("\n" + "=" * 50)
        print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"📊 {len(python_files)} fichiers analysés")
        print(f"📝 Logs enregistrés: logs/experiment_data.json")
        print("=" * 50)
        
    except ImportError as e:
        print(f"\n❌ ERREUR: Impossible d'importer l'agent Auditeur")
        print(f"   Message: {e}")
        print("\n📌 Assurez-vous que:")
        print("   1. Le fichier src/auditor_agent.py existe")
        print("   2. La classe AuditorAgent est bien définie")
        print("   3. src/utils/logger.py existe")
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()