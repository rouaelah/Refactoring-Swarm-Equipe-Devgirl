from google import genai
import os
from dotenv import load_dotenv

# Charger les variables d'environnement (.env doit contenir GOOGLE_API_KEY)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY manquante dans .env")

# Initialiser le client Gemini
client = genai.Client(api_key=api_key)

# Lister les modèles disponibles
print("✅ Modèles disponibles avec ta clé API :\n")
for model in client.models.list():
    print(f"- {model.name}")