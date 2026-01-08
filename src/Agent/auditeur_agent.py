"""
Auditor Agent (L'Agent Auditeur) - Version compatible langgraph==0.0.25

Responsable de :
1. Lire le code Python dans un dossier
2. Lancer l'analyse statique (avec pylint)
3. Produire un plan de refactoring structuré
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

# Import des outils internes
try:
    from src.tools.file_ops import (
        safe_read_file,
        safe_list_files,
        initialize_sandbox,
        FileOpsError
    )
    from src.utils.logger import log_experiment, ActionType
except ImportError as e:
    print(f"❌ Erreur d'import des modules internes: {e}")
    print("Assure-toi que le projet est correctement structuré.")
    sys.exit(1)

# Vérifier si LangGraph est disponible
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import Tool
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph disponible")
except ImportError as e:
    logger.warning(f"⚠️  LangGraph non disponible: {e}")
    logger.info("Utilisation de la version simplifiée de l'agent")


class AuditorAgent:
    """
    Agent spécialisé dans l'analyse de code Python et la génération de plans de refactoring.
    Compatible avec et sans LangGraph.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_iterations: int = 3
    ):
        """
        Initialise l'agent Auditeur.
        
        Args:
            model_name: Nom du modèle Gemini à utiliser
            temperature: Créativité du modèle (0.0 à 1.0)
            max_iterations: Nombre maximum d'itérations pour l'analyse
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialiser le modèle LLM
        self.llm = self._initialize_llm()
        
        # Initialiser les outils
        self.tools = self._initialize_tools()
        
        # Initialiser le système de prompts
        self.system_prompt = self._get_system_prompt()
        
        # État de l'agent
        self.current_target_dir = None
        self.analysis_results = {}
        
        logger.info(f"Auditor Agent initialized with model: {model_name}")
    
    def _initialize_llm(self):
        """
        Initialise le modèle LLM avec Gemini.
        """
        try:
            # Vérifier que la clé API est configurée
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            # Import conditionnel
            if LANGGRAPH_AVAILABLE:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    google_api_key=api_key
                )
            else:
                # Version simplifiée si LangGraph n'est pas disponible
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel(self.model_name)
            
            logger.debug(f"LLM initialized: {self.model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_tools(self) -> List[Dict]:
        """
        Initialise les outils disponibles pour l'agent.
        """
        tools = [
            {
                "name": "read_python_file",
                "func": self._tool_read_python_file,
                "description": "Lire le contenu d'un fichier Python"
            },
            {
                "name": "list_python_files",
                "func": self._tool_list_python_files,
                "description": "Lister tous les fichiers Python dans un répertoire"
            },
            {
                "name": "run_static_analysis",
                "func": self._tool_run_static_analysis,
                "description": "Exécuter l'analyse statique (pylint) sur un fichier Python"
            },
            {
                "name": "analyze_code_complexity",
                "func": self._tool_analyze_code_complexity,
                "description": "Analyser la complexité cyclomatique d'un fichier Python"
            },
            {
                "name": "check_test_coverage",
                "func": self._tool_check_test_coverage,
                "description": "Vérifier si un fichier a des tests associés"
            }
        ]
        
        logger.info(f"Initialized {len(tools)} tools for Auditor Agent")
        return tools
    
    def _get_system_prompt(self) -> str:
        """
        Retourne le prompt système pour l'agent Auditeur.
        """
        return """Tu es un Expert Python Senior spécialisé en analyse de code et refactoring.

# MISSION
Analyser le code Python pour identifier:
1. **Bugs et erreurs** de syntaxe ou logiques
2. **Problèmes de qualité**: violations PEP8, mauvaises pratiques
3. **Manques de documentation**: absence de docstrings, comments
4. **Problèmes de test**: absence de tests ou tests insuffisants
5. **Opportunités de refactoring**: code dupliqué, complexité excessive

# FORMAT DE SORTIE
Produis un plan de refactoring structuré avec:
- Résumé général
- Liste des problèmes par catégorie
- Priorité des corrections
- Estimation d'effort

# DIRECTIVES
- Sois précis et concret
- Donne des exemples de code problématique
- Propose des solutions spécifiques
- Classe les problèmes par priorité
"""
    
    def _tool_read_python_file(self, file_path: str) -> str:
        """
        Outil: Lire un fichier Python.
        """
        try:
            content = safe_read_file(file_path)
            
            # Log l'action
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "read_python_file",
                    "file_path": file_path,
                    "file_size": len(content),
                    "input_prompt": f"Read Python file: {file_path}",
                    "output_response": f"Successfully read {len(content)} characters"
                },
                status="SUCCESS"
            )
            
            return content
            
        except Exception as e:
            error_msg = f"Failed to read file {file_path}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "read_python_file",
                    "file_path": file_path,
                    "input_prompt": f"Read Python file: {file_path}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return f"ERROR: {error_msg}"
    
    def _tool_list_python_files(self, directory: str) -> List[str]:
        """
        Outil: Lister les fichiers Python dans un répertoire.
        """
        try:
            files = safe_list_files(directory, pattern="*.py", recursive=True)
            file_paths = [str(f) for f in files]
            
            # Log l'action
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "list_python_files",
                    "directory": directory,
                    "file_count": len(file_paths),
                    "input_prompt": f"List Python files in: {directory}",
                    "output_response": f"Found {len(file_paths)} Python files"
                },
                status="SUCCESS"
            )
            
            return file_paths
            
        except Exception as e:
            error_msg = f"Failed to list files in {directory}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "list_python_files",
                    "directory": directory,
                    "input_prompt": f"List Python files in: {directory}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return [f"ERROR: {error_msg}"]
    
    def _tool_run_static_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Outil: Exécuter pylint sur un fichier Python.
        """
        try:
            # Vérifier que le fichier existe
            if not Path(file_path).exists():
                return {"error": f"File does not exist: {file_path}"}
            
            # Exécuter pylint
            result = subprocess.run(
                ["pylint", "--output-format=json", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            analysis_result = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode in [0, 4, 8, 16, 24, 28]
            }
            
            # Parser le JSON de sortie
            if result.stdout and result.stdout.strip():
                try:
                    pylint_data = json.loads(result.stdout)
                    analysis_result["pylint_data"] = pylint_data
                    analysis_result["issue_count"] = len(pylint_data)
                except json.JSONDecodeError:
                    analysis_result["pylint_data"] = None
            
            # Log l'action
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "run_static_analysis",
                    "file_path": file_path,
                    "pylint_return_code": result.returncode,
                    "issue_count": analysis_result.get("issue_count", 0),
                    "input_prompt": f"Run pylint on: {file_path}",
                    "output_response": f"Pylint analysis completed with return code {result.returncode}"
                },
                status="SUCCESS" if analysis_result["success"] else "FAILURE"
            )
            
            return analysis_result
            
        except subprocess.TimeoutExpired:
            error_msg = f"Pylint timeout for {file_path}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "run_static_analysis",
                    "file_path": file_path,
                    "input_prompt": f"Run pylint on: {file_path}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {"error": error_msg, "success": False}
            
        except Exception as e:
            error_msg = f"Failed to run pylint on {file_path}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "run_static_analysis",
                    "file_path": file_path,
                    "input_prompt": f"Run pylint on: {file_path}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {"error": error_msg, "success": False}
    
    def _tool_analyze_code_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Outil: Analyser la complexité du code.
        """
        try:
            content = safe_read_file(file_path)
            
            # Analyse simplifiée
            complexity_indicators = {
                "line_count": len(content.split('\n')),
                "function_count": content.count("def "),
                "class_count": content.count("class "),
                "import_count": sum(1 for line in content.split('\n') if line.strip().startswith("import ")),
                "nested_loops": content.count("for ") + content.count("while "),
                "if_statements": content.count("if "),
                "try_blocks": content.count("try:"),
            }
            
            # Calcul d'un score de complexité
            complexity_score = (
                complexity_indicators["function_count"] * 2 +
                complexity_indicators["class_count"] * 3 +
                complexity_indicators["nested_loops"] * 1.5 +
                complexity_indicators["if_statements"] * 0.5
            )
            
            result = {
                **complexity_indicators,
                "complexity_score": round(complexity_score, 2),
                "complexity_level": "LOW" if complexity_score < 10 else "MEDIUM" if complexity_score < 25 else "HIGH"
            }
            
            # Log l'action
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "analyze_code_complexity",
                    "file_path": file_path,
                    "complexity_score": complexity_score,
                    "complexity_level": result["complexity_level"],
                    "input_prompt": f"Analyze code complexity of: {file_path}",
                    "output_response": f"Code complexity analysis completed. Score: {complexity_score}, Level: {result['complexity_level']}"
                },
                status="SUCCESS"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to analyze code complexity of {file_path}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "analyze_code_complexity",
                    "file_path": file_path,
                    "input_prompt": f"Analyze code complexity of: {file_path}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {"error": error_msg}
    
    def _tool_check_test_coverage(self, file_path: str) -> Dict[str, Any]:
        """
        Outil: Vérifier la présence de tests.
        """
        try:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            
            # Chercher des fichiers de test
            test_files = []
            for test_pattern in [f"test_{file_name}", f"{file_path_obj.stem}_test.py"]:
                test_path = file_path_obj.parent / test_pattern
                if test_path.exists():
                    test_files.append(str(test_path))
            
            # Vérifier dans un sous-dossier tests/
            tests_dir = file_path_obj.parent / "tests"
            if tests_dir.exists():
                for test_file in tests_dir.glob(f"*{file_path_obj.stem}*"):
                    if test_file.is_file() and test_file.suffix == ".py":
                        test_files.append(str(test_file))
            
            result = {
                "has_tests": len(test_files) > 0,
                "test_files": test_files,
                "test_count": len(test_files),
                "test_coverage": "PRESENT" if test_files else "MISSING"
            }
            
            # Log l'action
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "check_test_coverage",
                    "file_path": file_path,
                    "has_tests": result["has_tests"],
                    "test_count": result["test_count"],
                    "input_prompt": f"Check test coverage for: {file_path}",
                    "output_response": f"Test coverage check completed. Tests found: {result['test_count']}"
                },
                status="SUCCESS"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to check test coverage for {file_path}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "tool": "check_test_coverage",
                    "file_path": file_path,
                    "input_prompt": f"Check test coverage for: {file_path}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {"error": error_msg}
    
    def analyze_directory(self, target_dir: str) -> Dict[str, Any]:
        """
        Analyser un répertoire complet et produire un plan de refactoring.
        """
        try:
            logger.info(f"🔍 Début de l'analyse du répertoire: {target_dir}")
            self.current_target_dir = target_dir
            
            # Initialiser le sandbox
            initialize_sandbox()
            
            # 1. Lister les fichiers Python
            python_files = self._tool_list_python_files(target_dir)
            
            # Filtrer les erreurs
            valid_files = [f for f in python_files if isinstance(f, str) and not f.startswith("ERROR")]
            
            if not valid_files:
                error_msg = f"Aucun fichier Python valide trouvé dans {target_dir}"
                logger.error(error_msg)
                return {"error": error_msg, "success": False}
            
            logger.info(f"📁 {len(valid_files)} fichiers Python à analyser")
            
            # 2. Collecter les analyses
            file_analyses = {}
            for file_path in valid_files[:3]:  # Limiter à 3 fichiers pour les tests
                logger.info(f"  📄 Analyse de: {file_path}")
                
                content = self._tool_read_python_file(file_path)
                if isinstance(content, str) and content.startswith("ERROR"):
                    continue
                
                static_analysis = self._tool_run_static_analysis(file_path)
                complexity_analysis = self._tool_analyze_code_complexity(file_path)
                test_coverage = self._tool_check_test_coverage(file_path)
                
                file_analyses[file_path] = {
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "static_analysis": static_analysis,
                    "complexity": complexity_analysis,
                    "test_coverage": test_coverage
                }
            
            # 3. Générer le plan de refactoring
            refactoring_plan = self._generate_refactoring_plan(file_analyses, target_dir)
            
            # 4. Résultats finaux
            final_result = {
                "success": True,
                "target_directory": target_dir,
                "analysis_timestamp": datetime.now().isoformat(),
                "files_analyzed": list(file_analyses.keys()),
                "file_analyses": file_analyses,
                "refactoring_plan": refactoring_plan,
                "summary": {
                    "total_files_analyzed": len(file_analyses),
                    "files_with_issues": sum(1 for a in file_analyses.values() 
                                           if a.get("static_analysis", {}).get("issue_count", 0) > 0),
                    "files_without_tests": sum(1 for a in file_analyses.values() 
                                              if not a.get("test_coverage", {}).get("has_tests", True)),
                }
            }
            
            logger.info("✅ Analyse terminée avec succès")
            return final_result
            
        except Exception as e:
            error_msg = f"Échec de l'analyse du répertoire {target_dir}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "target_directory": target_dir,
                    "input_prompt": f"Analyze directory: {target_dir}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {
                "success": False,
                "error": error_msg,
                "target_directory": target_dir
            }
    
    def _generate_refactoring_plan(self, file_analyses: Dict, target_dir: str) -> Dict[str, Any]:
        """
        Générer un plan de refactoring avec le LLM.
        """
        try:
            # Préparer le contexte pour le LLM
            analysis_summary = []
            for file_path, analysis in file_analyses.items():
                file_name = Path(file_path).name
                issues = analysis.get("static_analysis", {}).get("issue_count", 0)
                has_tests = analysis.get("test_coverage", {}).get("has_tests", False)
                complexity = analysis.get("complexity", {}).get("complexity_level", "UNKNOWN")
                
                analysis_summary.append(
                    f"- {file_name}: {issues} problèmes, tests: {has_tests}, complexité: {complexity}"
                )
            
            # Créer le prompt
            prompt = f"""
{self.system_prompt}

# CONTEXTE
Répertoire analysé: {target_dir}
Nombre de fichiers analysés: {len(file_analyses)}

# RÉSUMÉ DES ANALYSES
{chr(10).join(analysis_summary)}

# TÂCHE
Génère un plan de refactoring structuré qui inclut:
1. Résumé général des problèmes
2. Catégories de problèmes (bugs, qualité, documentation, tests, refactoring)
3. Priorité des corrections (Haute/Moyenne/Basse)
4. Estimation du temps nécessaire

Format attendu (JSON):
{{
  "summary": "résumé ici",
  "issues_by_category": {{
    "bugs": ["problème1", "problème2"],
    "quality": ["problème1", "problème2"],
    "documentation": ["problème1", "problème2"],
    "testing": ["problème1", "problème2"],
    "refactoring_opportunities": ["problème1", "problème2"]
  }},
  "priority": ["high", "medium", "low"],
  "estimated_effort": "X heures"
}}
"""
            
            # Appeler le LLM
            if LANGGRAPH_AVAILABLE and hasattr(self.llm, 'invoke'):
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                response_text = response.content
            else:
                # Version simple
                response_text = self.llm.generate_content(prompt).text
            
            # Essayer d'extraire le JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Impossible de parser le JSON, utilisation du texte brut")
            
            # Retourner la réponse brute si pas de JSON
            return {
                "raw_response": response_text,
                "summary": "Plan généré par l'agent",
                "parsing_note": "Format JSON non détecté"
            }
            
        except Exception as e:
            logger.error(f"Erreur génération plan: {str(e)}")
            return {"error": f"Failed to generate plan: {str(e)}"}


# Version ultra-simplifiée (backup)
class SimpleAuditor:
    """Auditeur minimal pour tests rapides."""
    
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY manquant")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def quick_analyze(self, file_path: str) -> str:
        """Analyse rapide d'un fichier."""
        try:
            content = safe_read_file(file_path)
            
            prompt = f"""
Analyse ce code Python:
{content}

Liste:
1. Bugs potentiels
2. Problèmes de style PEP8
3. Manque de documentation
4. Suggestions d'amélioration
"""
            
            response = self.model.generate_content(prompt)
            
            log_experiment(
                agent_name="SimpleAuditor",
                model_used="gemini-1.5-flash",
                action=ActionType.ANALYSIS,
                details={
                    "file_path": file_path,
                    "input_prompt": prompt[:300] + "..." if len(prompt) > 300 else prompt,
                    "output_response": response.text[:500] + "..." if len(response.text) > 500 else response.text
                },
                status="SUCCESS"
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Erreur analyse simple: {str(e)}")
            return f"Erreur: {str(e)}"


# Fonction principale
def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Auditeur - Analyse de code Python")
    parser.add_argument("--target_dir", required=True, help="Répertoire à analyser")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Modèle Gemini")
    parser.add_argument("--verbose", action="store_true", help="Logs détaillés")
    parser.add_argument("--simple", action="store_true", help="Utiliser la version simple")
    
    args = parser.parse_args()
    
    # Configuration logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🧪 TEST AGENT AUDITEUR")
    print("=" * 60)
    
    try:
        if args.simple:
            print("🔧 Mode simple activé")
            auditor = SimpleAuditor()
            
            # Trouver un fichier Python
            target_path = Path(args.target_dir)
            python_files = list(target_path.glob("*.py"))
            
            if not python_files:
                print(f"❌ Aucun fichier .py dans {args.target_dir}")
                return
            
            first_file = python_files[0]
            print(f"📄 Analyse de: {first_file}")
            
            result = auditor.quick_analyze(str(first_file))
            print("\n📊 RÉSULTAT:")
            print("-" * 40)
            print(result)
            
        else:
            print("🚀 Mode complet avec AuditorAgent")
            auditor = AuditorAgent(model_name=args.model)
            
            print(f"📁 Analyse du répertoire: {args.target_dir}")
            result = auditor.analyze_directory(args.target_dir)
            
            if result.get("success"):
                print("\n✅ ANALYSE RÉUSSIE!")
                print(f"📊 Fichiers analysés: {len(result['files_analyzed'])}")
                
                plan = result.get("refactoring_plan", {})
                if "summary" in plan:
                    print(f"\n📋 Résumé: {plan['summary'][:200]}...")
                
                # Sauvegarder
                output_file = Path(args.target_dir) / "audit_report.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n💾 Rapport sauvegardé: {output_file}")
                
            else:
                print(f"\n❌ ERREUR: {result.get('error', 'Inconnue')}")
                
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()