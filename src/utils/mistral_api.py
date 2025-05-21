import os
import logging
from typing import List, Dict, Any, Optional
from mistralai import Mistral, UserMessage

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MistralAPIError(Exception):
    """Exception personnalisée pour les erreurs de l'API Mistral"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def create_ingredient_description(ingredients: List[Dict[str, Any]]) -> str:
    """
    Crée une description nutritionnelle concise pour un plat à partir de ses ingrédients.
    
    Args:
        ingredients (List[Dict[str, Any]]): Liste des ingrédients avec leurs quantités
    
    Returns:
        str: Description générée du plat
    
    Raises:
        MistralAPIError: En cas d'erreur avec l'API Mistral
    """
    api_key = "wVAoRHAtUHGiYcZSbd41opWKlUfOqY6j"
    if not api_key:
        raise ValueError("La clé API Mistral est manquante.")

    try:
        client = Mistral(api_key=api_key)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
        raise MistralAPIError("Échec de l'initialisation du client Mistral")

    model = "mistral-large-latest"

    ingredients_text = "\n".join([
        f"- {ing.get('name', 'Unknown')} ({ing.get('grams', 0):.1f}g)"
        for ing in ingredients
    ])

    prompt = f"""Create a simple dish description using these ingredients and their quantities:

{ingredients_text}

Generate a VERY short description (max 30 tokens) that:
1. Lists the main ingredients and their quantities
2. Uses simple and clear language
3. Avoids any nutritional analysis or suggestions
"""

    try:
        response = client.chat.complete(
            model=model,
            messages=[UserMessage(content=prompt)]
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower():
            raise MistralAPIError("Limite de requêtes dépassée", status_code=429)
        elif "Service tier capacity exceeded" in error_msg:
            raise MistralAPIError("Capacité du service dépassée", status_code=429)
        else:
            raise MistralAPIError(f"Erreur API Mistral : {error_msg}") 