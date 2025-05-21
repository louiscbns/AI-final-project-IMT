import torch
from nutrition_predictor import NutritionPredictor
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model():
    """
    Initialise le modèle de prédiction nutritionnelle avec des poids aléatoires.
    """
    # Création du dossier models s'il n'existe pas
    models_dir = Path("image_prediction/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialisation du modèle
    model = NutritionPredictor()
    
    # Initialisation des poids avec des valeurs aléatoires
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Sauvegarde du modèle
    model_path = models_dir / "nutrition_predictor.pth"
    model.save_model(str(model_path))
    logger.info(f"Modèle initialisé et sauvegardé à {model_path}")

if __name__ == "__main__":
    initialize_model() 