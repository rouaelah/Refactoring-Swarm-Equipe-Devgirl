# tester que les agents loggent correctement
# quality/test_integration.py
import subprocess
import sys
import os
import json
import tempfile

# Ajoute le dossier parent au chemin Python pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_logger_import():
    """Teste que le logger est correctement importable"""
    print("Test d'import du logger...")
    try:
        from src.utils.logger import log_experiment, ActionType
        print(" Logger importé correctement")
        return True
    except ImportError as e:
        print(f"Impossible d'importer le logger : {e}")
        return False

def test_logger_basic_usage():
    """Teste l'usage basique du logger"""
    print("\nTest d'écriture basique...")
    
    try:
        from src.utils.logger import log_experiment, ActionType
        
        # Sauvegarder l'ancien fichier de log s'il existe
        log_file = "logs/experiment_data.json"
        backup_exists = os.path.exists(log_file)
        
        if backup_exists:
            with open(log_file, 'r') as f:
                backup_data = json.load(f)
        
        # Créer un test
        log_experiment(
            agent_name="IntegrationTestAgent",
            model_used="test-model-1.0",
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Ceci est un prompt de test",
                "output_response": "Ceci est une réponse de test",
                "test_scenario": "integration_basic"
            },
            status="SUCCESS"
        )
        
        # Vérifier que le fichier existe
        if not os.path.exists(log_file):
            print(" Fichier de logs non créé")
            return False
            
        # Vérifier le contenu
        with open(log_file, 'r') as f:
            data = json.load(f)
            
        if len(data) == 0:
            print(" Aucune donnée dans le fichier")
            return False
            
        last_entry = data[-1]
        if last_entry.get("agent") != "IntegrationTestAgent":
            print(" Données incorrectes dans le fichier")
            return False
        
        print("Écriture dans les logs réussie")
        
        # Restaurer les données originales si nécessaire
        if backup_exists:
            with open(log_file, 'w') as f:
                json.dump(backup_data, f, indent=4)
        else:
            os.remove(log_file)
            
        return True
        
    except Exception as e:
        print(f" Échec de l'écriture : {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_action_types():
    """Teste tous les types d'actions possibles"""
    print("\n🔧 Test de tous les types d'actions...")
    
    try:
        from src.utils.logger import log_experiment, ActionType
        
        # Utiliser un fichier temporaire pour ne pas polluer les vrais logs
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name
        
        # Simuler le fichier de log
        import src.utils.logger as logger_module
        original_log_file = logger_module.LOG_FILE
        logger_module.LOG_FILE = tmp_path
        
        actions_to_test = [
            (ActionType.ANALYSIS, "Analyse de code"),
            (ActionType.GENERATION, "Génération de tests"),
            (ActionType.DEBUG, "Debug d'erreur"),
            (ActionType.FIX, "Correction de bug")
        ]
        
        for action_type, description in actions_to_test:
            log_experiment(
                agent_name="TestAgent",
                model_used="test-model",
                action=action_type,
                details={
                    "input_prompt": f"Prompt pour {description}",
                    "output_response": f"Réponse pour {description}",
                    "test": "yes"
                },
                status="SUCCESS"
            )
            print(f"   {action_type.value} - OK")
        
        # Vérifier que toutes les actions sont bien enregistrées
        with open(tmp_path, 'r') as f:
            data = json.load(f)
        
        if len(data) != len(actions_to_test):
            print(f" Nombre incorrect d'entrées: {len(data)} au lieu de {len(actions_to_test)}")
            logger_module.LOG_FILE = original_log_file
            os.unlink(tmp_path)
            return False
        
        # Restaurer le chemin original
        logger_module.LOG_FILE = original_log_file
        
        # Nettoyer
        os.unlink(tmp_path)
        
        print(" Tous les types d'actions fonctionnent")
        return True
        
    except Exception as e:
        print(f"Erreur lors du test des actions: {e}")
        return False

def test_missing_prompts_error():
    """Teste que le logger génère une erreur quand les prompts sont manquants"""
    print("\n Test d'erreur pour prompts manquants...")
    
    try:
        from src.utils.logger import log_experiment, ActionType
        
        # Essayer de logger sans les prompts obligatoires
        log_experiment(
            agent_name="TestAgent",
            model_used="test-model",
            action=ActionType.ANALYSIS,
            details={},  # Pas de prompts !
            status="SUCCESS"
        )
        
        print(" Le logger aurait dû lever une exception pour prompts manquants")
        return False
        
    except ValueError as e:
        if "input_prompt" in str(e) and "output_response" in str(e):
            print(" Le logger détecte correctement les prompts manquants")
            return True
        else:
            print(f" Mauvais message d'erreur: {e}")
            return False
    except Exception as e:
        print(f" Exception inattendue: {e}")
        return False

def test_complete_workflow():
    """Teste le workflow complet de logging"""
    print("\n" + "="*50)
    print(" TEST D'INTÉGRATION COMPLET DU LOGGING")
    print("="*50)
    
    tests = [
        ("Import du logger", test_logger_import),
        ("Usage basique", test_logger_basic_usage),
        ("Tous les types d'actions", test_all_action_types),
        ("Validation des prompts", test_missing_prompts_error),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\ Test: {test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f" Exception durant le test: {e}")
            results.append((test_name, False))
    
    # Affichage des résultats
    print("\n" + "="*50)
    print(" RÉSULTATS DES TESTS")
    print("="*50)
    
    all_passed = True
    for test_name, success in results:
        status = " PASSÉ" if success else "ÉCHOUÉ"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    # Vérification finale avec verify.py
    print("\n Validation finale avec verify.py...")
    try:
        # Importer directement la fonction
        from verify import verify_log_file
        verify_success = verify_log_file()
    except ImportError:
        # Si verify.py n'est pas importable, essayer de l'exécuter
        print("  Impossible d'importer verify.py, tentative d'exécution...")
        result = subprocess.run([sys.executable, "quality/verify.py"], 
                              capture_output=True, text=True)
        verify_success = "SUCCÈS" in result.stdout
    
    if verify_success:
        print(" verify.py confirme la validité des logs")
    else:
        print(" verify.py a trouvé des problèmes")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print(" Lancement des tests d'intégration du logging...")
    
    if test_complete_workflow():
        print("\n" + "" * 20)
        print("TOUS LES TESTS DE LOGGING PASSENT !")
        print("" * 20)
        print("\nLe système de logging est prêt pour la production.")
        sys.exit(0)
    else:
        print("\n" + "erreur" * 20)
        print("CERTAINS TESTS DE LOGGING ÉCHOUENT !")
        print("erreur" * 20)
        print("\nCorrige les problèmes avant de continuer.")
        sys.exit(1)