"""
LLM Client for Gemini API integration using LangChain.
"""
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.utils.logger import log_experiment, ActionType

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  langchain-google-genai not installed. It's in your requirements.txt")


class LangChainGeminiClient:
    """Client for Google Gemini API using LangChain"""

    # Table de correspondance pour corriger les noms obsolètes
    MODEL_ALIASES = {
        "gemini-pro": "gemini-1.5-pro",
        "gemini-flash": "gemini-1.5-flash",
        "gemini-1.0-pro": "gemini-1.0-pro-001",
    }

    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.1):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-google-genai package not installed")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        # Normaliser le nom du modèle
        model_name = self.MODEL_ALIASES.get(model_name, model_name)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=40,
        )
        self.model_name = model_name

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate response from Gemini using LangChain.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction

        Returns:
            Generated text
        """
        start_time = datetime.now()

        try:
            # Prepare messages
            messages = []

            if system_instruction:
                messages.append(SystemMessage(content=system_instruction))

            messages.append(HumanMessage(content=prompt))

            # Generate response
            response = self.llm.invoke(messages)

            # Extract text
            text = response.content if hasattr(response, "content") else str(response)

            # Log the generation
            log_experiment(
                agent_name="LLM_Client",
                model_used=self.model_name,
                action=ActionType.GENERATION,
                details={
                    "prompt_length": len(prompt),
                    "response_length": len(text),
                    "input_prompt": prompt[:500],  # Truncate for logs
                    "output_response": text[:500],  # Truncate for logs
                    "model": self.model_name,
                    "system_instruction": system_instruction[:200]
                    if system_instruction
                    else None,
                    "generation_time": (datetime.now() - start_time).total_seconds(),
                },
                status="SUCCESS",
            )

            return text

        except Exception as e:
            # Log the failure
            log_experiment(
                agent_name="LLM_Client",
                model_used=self.model_name,
                action=ActionType.GENERATION,
                details={
                    "input_prompt": prompt[:500],
                    "output_response": f"Error: {str(e)}",
                    "error": str(e),
                    "system_instruction": system_instruction[:200]
                    if system_instruction
                    else None,
                },
                status="FAILURE",
            )
            raise


# Factory function
def get_llm_client(model_name: str = "gemini-1.5-pro", temperature: float = 0.1):
    """
    Get LangChain Gemini client instance.

    Returns:
        LangChainGeminiClient instance
    """
    return LangChainGeminiClient(model_name=model_name, temperature=temperature)