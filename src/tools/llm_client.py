import os
import requests
import json
import time
import random
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logger import log_experiment, ActionType

load_dotenv()

class LangChainGeminiClient:
    """
    Client LLM ultra-robuste utilisant des requêtes HTTP directes.
    Optimisé pour les restrictions régionales et les quotas limités.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.1):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY manquante dans le fichier .env")
        
        # On utilise gemini-2.0-flash (présent dans votre liste de modèles supportés)
        self.model_name = "gemini-2.0-flash"
        self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent"
        self.temperature = temperature

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Génère une réponse avec une attente agressive pour éviter la 429."""
        
        # --- STRATÉGIE DE SURVIE QUOTA (20s + jitter) ---
        # On attend 20 secondes de base + un petit délai aléatoire pour que
        # si deux agents se lancent, ils ne frappent pas l'API exactement en même temps.
        wait_time = 20 + random.uniform(0, 5)
        print(f"⏳ Quota Protect : Attente de {wait_time:.2f}s avant l'appel API...")
        time.sleep(wait_time)
        
        start_time = datetime.now()
        
        # Construction du payload pour l'API Gemini
        payload = {
            "contents": [{
                "parts": [{"text": f"{system_instruction}\n\n{prompt}" if system_instruction else prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            # Appel API REST direct (v1 stable)
            response = requests.post(
                f"{self.url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            res_json = response.json()
            
            if response.status_code != 200:
                error_msg = res_json.get('error', {}).get('message', 'Erreur inconnue')
                if response.status_code == 429:
                    print("⚠️ Quota toujours atteint (429). Essayez d'attendre 2 minutes.")
                raise Exception(f"API Error {response.status_code}: {error_msg}")

            # Extraction sécurisée du texte
            try:
                text = res_json['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                text = "Erreur: Format de réponse API inattendu."
            
            # Log obligatoire pour la validation du TP (fichiers JSON)
            log_experiment(
                agent_name="LLM_Client",
                model_used=self.model_name,
                action=ActionType.GENERATION,
                status="SUCCESS",
                details={
                    "input_prompt": prompt[:200],
                    "output_response": text[:200],
                    "generation_time": (datetime.now() - start_time).total_seconds()
                }
            )
            return text
            
        except Exception as e:
            # Logging obligatoire même en cas d'échec pour le bot de correction
            log_experiment(
                agent_name="LLM_Client",
                model_used=self.model_name,
                action=ActionType.GENERATION,
                status="FAILURE",
                details={
                    "input_prompt": prompt[:200],
                    "output_response": f"ERREUR: {str(e)}"
                }
            )
            print(f"❌ Erreur lors de l'appel Gemini: {e}")
            raise

def get_llm_client(model_name: str = "gemini-2.0-flash", temperature: float = 0.1, use_mock: bool = False):
    """Factory function."""
    return LangChainGeminiClient(model_name=model_name, temperature=temperature)