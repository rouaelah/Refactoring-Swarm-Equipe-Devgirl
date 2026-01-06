import os
from src.utils.logger import log_experiment, ActionType

class AuditorAgent:
    """Agent qui analyse le code Python et produit un plan de refactoring"""
    
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
    
    def analyze_single_file(self, file_path: str) -> dict:
        """
        Analyse un seul fichier Python
        Retourne: {"file": chemin, "issues": liste des problèmes, "summary": str}
        """
        # 1. Vérifier si le fichier existe
        if not os.path.exists(file_path):
            return {
                "file": file_path, 
                "issues": [], 
                "error": "Fichier non trouvé",
                "summary": f"ERREUR: {os.path.basename(file_path)} introuvable"
            }
        
        # 2. Lire le contenu du fichier
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            lines_count = len(code_content.split('\n'))
        except Exception as e:
            return {
                "file": file_path, 
                "issues": [], 
                "error": f"Erreur lecture: {e}",
                "summary": f"ERREUR: Impossible de lire {os.path.basename(file_path)}"
            }
        
        # 3. Créer un prompt simple
        prompt = f"""Tu es un expert Python senior. Analyse ce code source et identifie TOUS les problèmes.

FICHIER: {os.path.basename(file_path)}
LIGNES: {lines_count}

CODE À ANALYSER:
```python
{code_content}
ANALYSE À EFFECTUER:

BUGS & ERREURS:

Y a-t-il des erreurs de syntaxe?

Y a-t-il des erreurs de logique?

Y a-t-il des erreurs d'exécution potentielles (ex: division par zéro)?

Gestion des exceptions?

QUALITÉ & STYLE:

Le code suit-il PEP8?

Noms de variables/fonctions clairs?

Indentation correcte?

Longueur des lignes (< 80 caractères)?

DOCUMENTATION:

Y a-t-il des docstrings?

Commentaires pertinents?

Documentation manquante?

STRUCTURE:

Fonctions trop longues?

Code dupliqué?

Complexité cyclomatique élevée?

PERFORMANCE & SÉCURITÉ:

Optimisations possibles?

Problèmes de sécurité (ex: injection)?

FORMAT DE RÉPONSE OBLIGATOIRE:

[LIGNE X] [CATÉGORIE] Description concise du problème
→ Suggestion: correction spécifique et concrète

CATÉGORIES: BUG, STYLE, DOCUMENTATION, STRUCTURE, PERFORMANCE, SECURITY

EXEMPLE:

[LIGNE 5] [BUG] Division par zéro possible
→ Suggestion: Ajouter vérification: if b == 0: return None ou raise ValueError

Ton analyse doit être précise, technique et actionable."""

        # 4. SIMULATION (à remplacer par vrai Gemini plus tard) 
        simulated_response = f"""Analyse complète de {os.path.basename(file_path)}:

- [LIGNE 1] [DOCUMENTATION] Pas de docstring pour le module
  → Suggestion: Ajouter ""Module avec des fonctions de calcul"" en début de fichier

- [LIGNE 5] [BUG] Division par zéro possible sans vérification
  → Suggestion: if b == 0: raise ValueError("Division par zéro interdite")

- [LIGNE 8] [STYLE] Noms de variables trop courts ('x', 'y')
  → Suggestion: Remplacer par 'dividend', 'divisor'

- [LIGNE 12] [STRUCTURE] Fonction trop longue (15 lignes)
  → Suggestion: Diviser en sous-fonctions de max 10 lignes

RÉSUMÉ:
• Bugs critiques: 1
• Problèmes de style: 2  
• Documentation manquante: 1
• TOTAL PROBLÈMES: 4"""
        
        try:
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "file_analyzed": file_path,  # <-- ICI file_path EST DÉFINI
                    "file_size_bytes": len(code_content),
                    "lines_of_code": lines_count,
                    "input_prompt": prompt,  # OBLIGATOIRE
                    "output_response": simulated_response,  # OBLIGATOIRE
                    "simulation_mode": True  # À ENLEVER plus tard
                },
                status="SUCCESS"
            )
        except Exception as e:
            print(f"⚠️  Erreur logging: {e}")
            # Continuer même si logging échoue
    
        # 6. Retourner le résultat
        return {
            "file": file_path,
            "filename": os.path.basename(file_path),
            "issues": simulated_response,
            "lines_analyzed": lines_count,
            "summary": f"Analyse terminée: {os.path.basename(file_path)} ({lines_count} lignes, 4 problèmes détectés)"
        }