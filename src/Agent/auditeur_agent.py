"""
Auditor Agent (L'Agent Auditeur)

Responsable de :
1. Lire le code Python dans un dossier
2. Lancer l'analyse statique (avec pylint)
3. Produire un plan de refactoring structuré

Cet agent est conçu pour être utilisé avec LangGraph.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

# Import des outils internes
from src.tools.file_ops import (
    safe_read_file,
    safe_list_files,
    initialize_sandbox,
    FileOpsError
)
from src.utils.logger import log_experiment, ActionType

# Import LangGraph pour l'agent
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint import MemorySaver
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import Tool
except ImportError as e:
    logger.error(f"LangGraph or dependencies not installed: {e}")
    logger.error("Install with: pip install langgraph langchain-google-genai")
    raise


class AuditorAgent:
    """
    Agent spécialisé dans l'analyse de code Python et la génération de plans de refactoring.
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
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initialise le modèle LLM avec Gemini.
        """
        try:
            # Vérifier que la clé API est configurée
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key
            )
            
            # Test simple de connexion
            test_response = llm.invoke("Hello")
            logger.debug(f"LLM test response: {test_response.content[:50]}...")
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_tools(self) -> List[Tool]:
        """
        Initialise les outils disponibles pour l'agent.
        """
        tools = [
            Tool(
                name="read_python_file",
                func=self._tool_read_python_file,
                description="Lire le contenu d'un fichier Python. Args: file_path (str): Chemin du fichier"
            ),
            Tool(
                name="list_python_files",
                func=self._tool_list_python_files,
                description="Lister tous les fichiers Python dans un répertoire. Args: directory (str): Répertoire à analyser"
            ),
            Tool(
                name="run_static_analysis",
                func=self._tool_run_static_analysis,
                description="Exécuter l'analyse statique (pylint) sur un fichier Python. Args: file_path (str): Chemin du fichier"
            ),
            Tool(
                name="analyze_code_complexity",
                func=self._tool_analyze_code_complexity,
                description="Analyser la complexité cyclomatique d'un fichier Python. Args: file_path (str): Chemin du fichier"
            ),
            Tool(
                name="check_test_coverage",
                func=self._tool_check_test_coverage,
                description="Vérifier si un fichier a des tests associés. Args: file_path (str): Chemin du fichier"
            )
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
Tu dois produire un plan de refactoring structuré en JSON avec:
{
  "summary": "Résumé général des problèmes",
  "files_analyzed": ["file1.py", "file2.py"],
  "issues_by_category": {
    "bugs": ["description1", "description2"],
    "quality": ["description1", "description2"],
    "documentation": ["description1", "description2"],
    "testing": ["description1", "description2"],
    "refactoring_opportunities": ["description1", "description2"]
  },
  "priority": ["high", "medium", "low"],  # Priorité des corrections
  "estimated_effort": "1-2 heures"  # Estimation du temps de correction
}

# DIRECTIVES
- Sois précis et concret
- Donne des exemples de code problématique
- Propose des solutions spécifiques
- Évalue la complexité des corrections
- Classe les problèmes par priorité

# RESTRICTIONS
- Ne modifie PAS le code toi-même
- Ne sors pas du répertoire sandbox
- Utilise uniquement les outils fournis
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
                timeout=30  # Timeout de 30 secondes
            )
            
            analysis_result = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode in [0, 4, 8, 16, 24, 28]  # Codes de sortie pylint acceptables
            }
            
            # Parser le JSON de sortie si disponible
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
        Outil: Analyser la complexité du code (simplifié).
        """
        try:
            content = safe_read_file(file_path)
            
            # Analyse simplifiée de la complexité
            lines = content.split('\n')
            complexity_indicators = {
                "line_count": len(lines),
                "function_count": content.count("def "),
                "class_count": content.count("class "),
                "import_count": sum(1 for line in lines if line.strip().startswith("import ")),
                "nested_loops": content.count("for ") + content.count("while "),
                "if_statements": content.count("if "),
                "try_blocks": content.count("try:"),
            }
            
            # Calcul d'un score de complexité simple
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
            file_dir = file_path_obj.parent
            
            # Chercher des fichiers de test
            test_files = []
            for test_pattern in [f"test_{file_name}", f"{file_path_obj.stem}_test.py"]:
                test_path = file_dir / test_pattern
                if test_path.exists():
                    test_files.append(str(test_path))
            
            # Vérifier aussi dans un sous-dossier tests/
            tests_dir = file_dir / "tests"
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
        
        Args:
            target_dir: Répertoire contenant le code Python à analyser
            
        Returns:
            Plan de refactoring structuré
        """
        try:
            logger.info(f"Starting analysis of directory: {target_dir}")
            self.current_target_dir = target_dir
            
            # Initialiser le sandbox si nécessaire
            initialize_sandbox()
            
            # 1. Lister tous les fichiers Python
            python_files = self._tool_list_python_files(target_dir)
            
            if not python_files or (len(python_files) == 1 and python_files[0].startswith("ERROR")):
                error_msg = f"No Python files found in {target_dir} or error listing files"
                logger.error(error_msg)
                return {"error": error_msg, "files_analyzed": [], "success": False}
            
            logger.info(f"Found {len(python_files)} Python files to analyze")
            
            # 2. Collecter les analyses pour chaque fichier
            file_analyses = {}
            for file_path in python_files:
                if not isinstance(file_path, str) or file_path.startswith("ERROR"):
                    continue
                    
                logger.debug(f"Analyzing file: {file_path}")
                
                # Lire le contenu
                content = self._tool_read_python_file(file_path)
                if isinstance(content, str) and content.startswith("ERROR"):
                    continue
                
                # Analyse statique avec pylint
                static_analysis = self._tool_run_static_analysis(file_path)
                
                # Analyse de complexité
                complexity_analysis = self._tool_analyze_code_complexity(file_path)
                
                # Vérification des tests
                test_coverage = self._tool_check_test_coverage(file_path)
                
                file_analyses[file_path] = {
                    "content": content[:1000] + "..." if len(content) > 1000 else content,
                    "static_analysis": static_analysis,
                    "complexity": complexity_analysis,
                    "test_coverage": test_coverage
                }
            
            # 3. Préparer le contexte pour le LLM
            analysis_context = {
                "directory": target_dir,
                "total_files": len(python_files),
                "file_analyses": file_analyses,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # 4. Appeler le LLM pour générer le plan de refactoring
            refactoring_plan = self._generate_refactoring_plan(analysis_context)
            
            # 5. Structurer les résultats finaux
            final_result = {
                "success": True,
                "target_directory": target_dir,
                "analysis_timestamp": datetime.now().isoformat(),
                "files_analyzed": list(file_analyses.keys()),
                "file_analyses": file_analyses,
                "refactoring_plan": refactoring_plan,
                "summary": {
                    "total_files": len(python_files),
                    "files_with_issues": sum(1 for analysis in file_analyses.values() 
                                           if analysis.get("static_analysis", {}).get("issue_count", 0) > 0),
                    "files_without_tests": sum(1 for analysis in file_analyses.values() 
                                              if not analysis.get("test_coverage", {}).get("has_tests", True)),
                    "high_complexity_files": sum(1 for analysis in file_analyses.values() 
                                                if analysis.get("complexity", {}).get("complexity_level") == "HIGH")
                }
            }
            
            logger.info(f"Analysis completed successfully for {target_dir}")
            
            # Log l'analyse complète
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "target_directory": target_dir,
                    "total_files": len(python_files),
                    "refactoring_plan_generated": bool(refactoring_plan),
                    "input_prompt": f"Analyze Python code in directory: {target_dir}",
                    "output_response": json.dumps(refactoring_plan, indent=2) if refactoring_plan else "No refactoring plan generated"
                },
                status="SUCCESS"
            )
            
            return final_result
            
        except Exception as e:
            error_msg = f"Failed to analyze directory {target_dir}: {str(e)}"
            logger.error(error_msg)
            
            log_experiment(
                agent_name="Auditor_Agent",
                model_used=self.model_name,
                action=ActionType.ANALYSIS,
                details={
                    "target_directory": target_dir,
                    "input_prompt": f"Analyze Python code in directory: {target_dir}",
                    "output_response": f"Error: {error_msg}"
                },
                status="FAILURE"
            )
            
            return {
                "success": False,
                "error": error_msg,
                "target_directory": target_dir
            }
    
    def _generate_refactoring_plan(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Générer un plan de refactoring avec le LLM.
        """
        try:
            # Préparer le prompt
            prompt = f"""
{self.system_prompt}

# CONTEXTE D'ANALYSE
Répertoire analysé: {analysis_context['directory']}
Nombre total de fichiers: {analysis_context['total_files']}
Timestamp: {analysis_context['analysis_timestamp']}

# RÉSULTATS D'ANALYSE
Voici les analyses détaillées pour chaque fichier:

{json.dumps(analysis_context['file_analyses'], indent=2, default=str)}

# TÂCHE
Génère un plan de refactoring structuré en JSON selon le format demandé.
Soyez spécifique, concret et proposez des solutions réalisables.
"""
            
            # Appeler le LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Essayer d'extraire le JSON de la réponse
            response_text = response.content
            
            # Chercher du JSON dans la réponse
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    refactoring_plan = json.loads(json_str)
                except json.JSONDecodeError:
                    # Si le JSON est invalide, utiliser le texte brut
                    refactoring_plan = {
                        "raw_response": response_text,
                        "summary": "Generated from LLM response",
                        "parsing_error": "Could not parse JSON from response"
                    }
            else:
                refactoring_plan = {
                    "raw_response": response_text,
                    "summary": "Generated from LLM response",
                    "parsing_note": "No JSON structure found in response"
                }
            
            return refactoring_plan
            
        except Exception as e:
            error_msg = f"Failed to generate refactoring plan: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "plan_generation_failed": True}
    
    def get_agent_graph(self) -> StateGraph:
        """
        Crée et retourne le graphe d'exécution LangGraph pour cet agent.
        
        Returns:
            StateGraph configuré pour l'agent Auditeur
        """
        # Définir l'état du graphe
        from typing import TypedDict, Annotated
        import operator
        
        class AgentState(TypedDict):
            """État de l'agent Auditeur."""
            target_dir: str
            python_files: List[str]
            current_file: str
            file_analyses: Dict[str, Dict]
            refactoring_plan: Dict[str, Any]
            iteration: int
            status: str  # "analyzing", "generating_plan", "completed", "error"
            error_message: Optional[str]
        
        # Créer le graphe
        graph_builder = StateGraph(AgentState)
        
        # Définir les nœuds
        def list_files_node(state: AgentState) -> AgentState:
            """Nœud: Lister les fichiers Python."""
            try:
                files = self._tool_list_python_files(state["target_dir"])
                return {
                    **state,
                    "python_files": files,
                    "status": "analyzing",
                    "iteration": state.get("iteration", 0) + 1
                }
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Failed to list files: {str(e)}"
                }
        
        def analyze_file_node(state: AgentState) -> AgentState:
            """Nœud: Analyser un fichier."""
            if not state["python_files"]:
                return {**state, "status": "completed"}
            
            current_file = state["python_files"][0]
            remaining_files = state["python_files"][1:]
            
            try:
                # Analyser le fichier
                content = self._tool_read_python_file(current_file)
                static_analysis = self._tool_run_static_analysis(current_file)
                complexity = self._tool_analyze_code_complexity(current_file)
                test_coverage = self._tool_check_test_coverage(current_file)
                
                file_analysis = {
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "static_analysis": static_analysis,
                    "complexity": complexity,
                    "test_coverage": test_coverage
                }
                
                # Mettre à jour les analyses
                current_analyses = state.get("file_analyses", {})
                current_analyses[current_file] = file_analysis
                
                return {
                    **state,
                    "python_files": remaining_files,
                    "current_file": current_file,
                    "file_analyses": current_analyses,
                    "status": "analyzing" if remaining_files else "generating_plan",
                    "iteration": state.get("iteration", 0) + 1
                }
                
            except Exception as e:
                logger.error(f"Failed to analyze file {current_file}: {e}")
                return {
                    **state,
                    "python_files": remaining_files,
                    "status": "analyzing" if remaining_files else "generating_plan",
                    "iteration": state.get("iteration", 0) + 1
                }
        
        def generate_plan_node(state: AgentState) -> AgentState:
            """Nœud: Générer le plan de refactoring."""
            try:
                analysis_context = {
                    "directory": state["target_dir"],
                    "total_files": len(state["file_analyses"]),
                    "file_analyses": state["file_analyses"]
                }
                
                refactoring_plan = self._generate_refactoring_plan(analysis_context)
                
                return {
                    **state,
                    "refactoring_plan": refactoring_plan,
                    "status": "completed",
                    "iteration": state.get("iteration", 0) + 1
                }
                
            except Exception as e:
                return {
                    **state,
                    "status": "error",
                    "error_message": f"Failed to generate plan: {str(e)}"
                }
        
        # Ajouter les nœuds au graphe
        graph_builder.add_node("list_files", list_files_node)
        graph_builder.add_node("analyze_file", analyze_file_node)
        graph_builder.add_node("generate_plan", generate_plan_node)
        
        # Définir le flux de contrôle
        graph_builder.set_entry_point("list_files")
        
        # Conditions de transition
        def should_analyze_files(state: AgentState) -> str:
            """Détermine s'il faut analyser d'autres fichiers."""
            if state["status"] == "error":
                return "error"
            elif state["python_files"]:
                return "analyze_file"
            else:
                return "generate_plan"
        
        def should_generate_plan(state: AgentState) -> str:
            """Détermine s'il faut générer le plan."""
            if state["status"] == "error":
                return "error"
            else:
                return "generate_plan"
        
        # Ajouter les arêtes conditionnelles
        graph_builder.add_conditional_edges(
            "list_files",
            should_analyze_files,
            {
                "analyze_file": "analyze_file",
                "generate_plan": "generate_plan",
                "error": END
            }
        )
        
        graph_builder.add_conditional_edges(
            "analyze_file",
            should_analyze_files,
            {
                "analyze_file": "analyze_file",
                "generate_plan": "generate_plan",
                "error": END
            }
        )
        
        graph_builder.add_edge("generate_plan", END)
        
        # Compiler le graphe
        graph = graph_builder.compile()
        
        return graph


# Fonction principale pour tester l'agent
def main():
    """
    Fonction principale pour tester l'agent Auditeur.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Auditor Agent - Analyse de code Python")
    parser.add_argument("--target_dir", required=True, help="Répertoire contenant le code Python à analyser")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Modèle Gemini à utiliser")
    parser.add_argument("--verbose", action="store_true", help="Activer les logs détaillés")
    
    args = parser.parse_args()
    
    # Configurer le logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Créer l'agent
        agent = AuditorAgent(model_name=args.model)
        
        # Analyser le répertoire
        result = agent.analyze_directory(args.target_dir)
        
        # Afficher les résultats
        if result.get("success"):
            print("\n" + "="*80)
            print("ANALYSE TERMINÉE AVEC SUCCÈS")
            print("="*80)
            
            print(f"\nRépertoire analysé: {result['target_directory']}")
            print(f"Fichiers analysés: {len(result['files_analyzed'])}")
            print(f"Timestamp: {result['analysis_timestamp']}")
            
            print("\n" + "-"*80)
            print("RÉSUMÉ:")
            print("-"*80)
            summary = result.get("summary", {})
            print(f"• Fichiers avec problèmes: {summary.get('files_with_issues', 0)}")
            print(f"• Fichiers sans tests: {summary.get('files_without_tests', 0)}")
            print(f"• Fichiers haute complexité: {summary.get('high_complexity_files', 0)}")
            
            print("\n" + "-"*80)
            print("PLAN DE REFACTORING:")
            print("-"*80)
            plan = result.get("refactoring_plan", {})
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            
            # Sauvegarder le rapport complet
            output_file = Path(args.target_dir) / "auditor_report.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nRapport complet sauvegardé dans: {output_file}")
            
        else:
            print(f"\nERREUR: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Erreur lors de l'exécution de l'agent: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()