import json
import time
import logging
from typing import List, Dict, Any, Optional
from .load_metadata import load_metadata
from ..utils.mistral_api import create_ingredient_description, MistralAPIError

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_metadata_modified(metadata: List[Dict[str, Any]], max_retries: int = 5, wait_seconds: int = 5, output_file_path: str = None) -> Optional[List[Dict[str, Any]]]:
    """
    Traite les métadonnées en ajoutant des descriptions générées par l'API Mistral.
    
    Args:
        metadata (List[Dict[str, Any]]): Liste des métadonnées à traiter
        max_retries (int): Nombre maximum de tentatives en cas d'erreur
        wait_seconds (int): Temps d'attente entre les tentatives
        output_file_path (str): Chemin pour sauvegarder les résultats intermédiaires
    
    Returns:
        Optional[List[Dict[str, Any]]]: Liste des métadonnées modifiées
    """
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