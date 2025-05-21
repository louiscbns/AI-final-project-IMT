import logging
import json
import numpy as np
from utils.embeddings import load_modified_metadata

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_data(file_path: str) -> tuple:
    """
    Charge les données du modèle (embeddings et labels).
    
    Args:
        file_path (str): Chemin vers le fichier embeddings.json
    
    Returns:
        tuple: (embeddings, labels, metadata)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            embeddings = np.array(data['embeddings'])
            labels = np.array(data['labels'])
            metadata = data['metadata']
            return embeddings, labels, metadata
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données du modèle : {e}")
        return None, None, None

def main():
    """
    Fonction principale qui charge et utilise les données déjà traitées.
    """
    print("\n🚀 Chargement des données traitées")

    # Chemins des fichiers
    metadata_path = 'data/food-nutrients/metadata_modified.json'
    embeddings_path = 'data/food-nutrients/embeddings.json'

    try:
        # 1. Charger les métadonnées modifiées
        metadata = load_modified_metadata(metadata_path)
        if metadata is None:
            return

        # 2. Charger les données du modèle
        embeddings, labels, model_metadata = load_model_data(embeddings_path)
        if embeddings is None or labels is None:
            return

        # Afficher les informations sur les données chargées
        print("\n📊 Informations sur les données chargées:")
        print(f"Nombre de plats: {len(metadata)}")
        print(f"Dimensions des embeddings: {embeddings.shape}")
        print(f"Dimensions des labels: {labels.shape}")
        print("\nCaractéristiques nutritionnelles:")
        for i, feature in enumerate(model_metadata['feature_names']):
            print(f"- {feature}:")
            print(f"  Moyenne: {np.mean(labels[:, i]):.2f}")
            print(f"  Écart-type: {np.std(labels[:, i]):.2f}")

        # Ici, vous pouvez ajouter votre code pour utiliser ces données
        # Par exemple, pour l'entraînement d'un modèle ou des prédictions

    except Exception as e:
        logger.error(f"Erreur lors du traitement : {str(e)}")
        raise

if __name__ == "__main__":
    main() 