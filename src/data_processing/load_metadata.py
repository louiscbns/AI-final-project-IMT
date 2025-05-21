import json
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_metadata(file_path, num_samples=None):
    """
    Charge les métadonnées depuis un fichier JSONL.
    
    Args:
        file_path (str): Chemin vers le fichier JSONL
        num_samples (int, optional): Nombre d'échantillons à charger. Si None, charge tous les échantillons.
    
    Returns:
        list: Liste des plats avec leurs métadonnées
    """
    dishes_data = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                data = json.loads(line)
                dishes_data.append({
                    'id': data.get('id'),
                    'split': data.get('split'),
                    'ingredients': data.get('ingredients', []),
                    'total_calories': data.get('total_calories'),
                    'total_mass': data.get('total_mass'),
                    'total_fat': data.get('total_fat'),
                    'total_carb': data.get('total_carb'),
                    'total_protein': data.get('total_protein')
                })
    except FileNotFoundError:
        logger.error(f"Erreur : Le fichier {file_path} est introuvable.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de décodage JSON dans le fichier {file_path} : {e}")
        return None
    except Exception as e:
        logger.error(f"Une erreur inattendue s'est produite lors du chargement : {e}")
        return None
    return dishes_data 