from IPython import get_ipython
from IPython.display import display
# %%
# Google Drive
from google.colab import drive
drive.mount('/content/drive/')
file_path ='/content/drive/MyDrive/Projet-IMT/metadata.jsonl'

# Imports
import json
import os
import torch
import ssl

!pip install mistralai
from mistralai import Mistral, UserMessage


!pip install ratelimit backoff



# Importez les autres librairies si elles sont utilisées par la suite
from transformers import AutoTokenizer, AutoModel





# %%
# Importer les dépendances nécessaires (Cette ligne n'est pas une instruction Python, elle peut être supprimée ou remplacée par des commentaires)

def load_metadata(file_path, num_samples=None):
    dishes_data = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                data = json.loads(line)
                # Assurez-vous que toutes les clés nécessaires existent dans le JSON
                # sinon ajoutez une gestion des erreurs ou des valeurs par défaut
                dishes_data.append({
                    'id': data.get('id'),
                    'split': data.get('split'),
                    'ingredients': data.get('ingredients', []), # Valeur par défaut pour ingredients si manquante
                    'total_calories': data.get('total_calories'),
                    'total_mass': data.get('total_mass'),
                    'total_fat': data.get('total_fat'),
                    'total_carb': data.get('total_carb'),
                    'total_protein': data.get('total_protein')
                })
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return None # Ou levez une exception
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {file_path} : {e}")
        return None
    except Exception as e:
        print(f"Une erreur inattendue s'est produite lors du chargement : {e}")
        return None
    return dishes_data




import time
import logging
from typing import List, Dict, Any, Optional

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

def create_ingredient_description(ingredients):
    """
    Crée une description nutritionnelle concise pour un plat à partir de ses ingrédients.
    Compatible avec le client Mistral >=1.0.0.
    """
    import os
    from mistralai import Mistral, UserMessage

    # 🔐 Récupérer l'API key Mistral (modifie ici si besoin)
    api_key = "wVAoRHAtUHGiYcZSbd41opWKlUfOqY6j"
    if not api_key:
        raise ValueError("La clé API Mistral est manquante. Définis-la avec os.environ ou dans le code.")

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




from typing import List, Dict, Any, Optional

def create_metadata_modified(metadata: List[Dict[str, Any]], max_retries: int = 5, wait_seconds: int = 5, output_file_path: str = None) -> Optional[List[Dict[str, Any]]]:
    if metadata is None:
        logger.error("Les métadonnées sont nulles")
        return None

    metadata_modified = []
    total_dishes = len(metadata)
    save_interval = 50  # Sauvegarder tous les 50 plats
    
    print(f"\n{'='*50}")
    print(f"Début du traitement de {total_dishes} plats")
    print(f"{'='*50}\n")
    
    for index, dish in enumerate(metadata, start=1):
        dish_id = dish.get('id', 'Unknown')
        print(f"\n[{index}/{total_dishes}] Traitement du plat ID: {dish_id}")
        print(f"{'-'*30}")
        
        logger.info(f"Traitement du plat {index}/{total_dishes} (ID {dish_id})")

        ingredients = dish.get('ingredients', [])
        if not isinstance(ingredients, list):
            logger.warning(f"Données d'ingrédients invalides pour le plat ID {dish_id}")
            description = "Description non disponible (données d'ingrédients invalides)."
        else:
            retries = 0
            while retries <= max_retries:
                try:
                    description = create_ingredient_description(ingredients)
                    break
                except MistralAPIError as e:
                    if e.status_code == 429:
                        retries += 1
                        if retries <= max_retries:
                            print(f"⚠️  Tentative {retries}/{max_retries} - Attente de {wait_seconds}s...")
                            logger.warning(f"Tentative {retries}/{max_retries} pour le plat ID {dish_id}. Attente de {wait_seconds}s...")
                            time.sleep(wait_seconds)
                        else:
                            print(f"❌ Échec après {max_retries} tentatives")
                            logger.error(f"Échec après {max_retries} tentatives pour le plat ID {dish_id}")
                            description = "Description non disponible (limite de requêtes dépassée)."
                    else:
                        print(f"❌ Erreur API: {e.message}")
                        logger.error(f"Erreur API pour le plat ID {dish_id}: {e.message}")
                        description = "Description non disponible (erreur API)."
                        break
                except Exception as e:
                    print(f"❌ Erreur inattendue: {str(e)}")
                    logger.error(f"Erreur inattendue pour le plat ID {dish_id}: {str(e)}")
                    description = "Description non disponible (erreur inattendue)."
                    break

        modified_dish = dish.copy()
        modified_dish['description'] = description
        metadata_modified.append(modified_dish)
        
        print(f"✅ Description créée : {description}")
        logger.info(f"Plat ID {dish_id} - Description créée : {description}")

        # Sauvegarder tous les 50 plats
        if output_file_path and index % save_interval == 0:
            try:
                with open(output_file_path, 'w') as f:
                    json.dump(metadata_modified, f, indent=4, ensure_ascii=False)
                print(f"\n💾 Sauvegarde intermédiaire effectuée ({index}/{total_dishes} plats)")
                logger.info(f"Sauvegarde intermédiaire effectuée après {index} plats")
            except Exception as e:
                print(f"❌ Erreur lors de la sauvegarde intermédiaire : {str(e)}")
                logger.error(f"Erreur lors de la sauvegarde intermédiaire : {str(e)}")

    print(f"\n{'='*50}")
    print(f"Traitement terminé : {len(metadata_modified)}/{total_dishes} plats traités")
    print(f"{'='*50}\n")

    return metadata_modified

# Charger les données brutes
file_path = '/content/drive/MyDrive/Projet-IMT/metadata.jsonl'
output_file_path = '/content/drive/MyDrive/Projet-IMT/metadata_modified.json'

try:
    raw_metadata = load_metadata(file_path)
    if raw_metadata is None:
        logger.error("Échec du chargement des métadonnées brutes")
        raise ValueError("Les métadonnées brutes sont nulles")

    # Traiter tous les plats avec sauvegarde intermédiaire
    metadata_with_descriptions = create_metadata_modified(
        raw_metadata,
        output_file_path=output_file_path
    )

    if metadata_with_descriptions is not None:
        try:
            with open(output_file_path, 'w') as f:
                json.dump(metadata_with_descriptions, f, indent=4, ensure_ascii=False)
            logger.info(f"Fichier metadata_modified.json créé à : {output_file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du fichier : {str(e)}")
            raise
    else:
        logger.error("La création des descriptions a échoué")
        raise ValueError("Échec de la création des descriptions")

except Exception as e:
    logger.error(f"Erreur lors du traitement : {str(e)}")
    raise


