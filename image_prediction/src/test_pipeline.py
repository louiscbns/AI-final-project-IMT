import os
from pathlib import Path
import logging
from pipeline import FoodImageAnalyzer
from segmentation import FoodSegmentation
from nutrition_predictor import NutritionPredictor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_colored_mask(mask, num_classes=5):
    """
    Crée un masque coloré pour la visualisation.
    
    Args:
        mask (numpy.ndarray): Masque de segmentation
        num_classes (int): Nombre de classes
        
    Returns:
        numpy.ndarray: Masque coloré
    """
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    colored_mask = np.zeros((*mask.shape, 3))
    
    for i in range(num_classes):
        colored_mask[mask == i] = colors[i][:3]
    
    return colored_mask

def visualize_results(image_path, segmentation_mask, nutrition_values, detected_ingredients):
    """
    Visualise les résultats de l'analyse.
    
    Args:
        image_path (str): Chemin vers l'image
        segmentation_mask (numpy.ndarray): Masque de segmentation
        nutrition_values (dict): Valeurs nutritionnelles prédites
        detected_ingredients (list): Liste des ingrédients détectés
    """
    # Création de la figure
    plt.figure(figsize=(15, 7))
    
    # Affichage de l'image originale
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title('Image Originale')
    plt.axis('off')
    
    # Création et affichage du masque coloré
    plt.subplot(1, 2, 2)
    colored_mask = create_colored_mask(segmentation_mask)
    plt.imshow(colored_mask)
    plt.title('Segmentation')
    plt.axis('off')
    
    # Ajout des informations nutritionnelles et des ingrédients
    info_text = "Ingrédients détectés:\n" + "\n".join(detected_ingredients)
    info_text += "\n\nValeurs nutritionnelles:\n"
    info_text += "\n".join([f"{k}: {v:.2f}" for k, v in nutrition_values.items()])
    
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    # Chemins des fichiers
    data_dir = Path("data/food-nutrients/test")
    
    # Initialisation des modèles
    analyzer = FoodImageAnalyzer()
    segmenter = FoodSegmentation()
    nutrition_predictor = NutritionPredictor()
    
    # Chargement du modèle entraîné
    model_path = Path("image_prediction/models/nutrition_predictor.pth")
    if model_path.exists():
        nutrition_predictor.load_model(str(model_path))
    
    # Récupération des 5 premières images
    image_files = sorted(list(data_dir.glob("*.png")))[:5]
    
    for image_path in image_files:
        logger.info(f"Analyse de l'image: {image_path.name}")
        
        try:
            # Segmentation
            segmentation_results = segmenter.segment_image(str(image_path))
            
            # Extraction des features et prédiction nutritionnelle
            image_tensor = analyzer.preprocess_image(str(image_path))
            features = analyzer.extract_features(image_tensor)
            nutrition_values = nutrition_predictor.predict(features)
            
            # Visualisation des résultats
            visualize_results(
                str(image_path),
                segmentation_results["segmentation_mask"],
                nutrition_values,
                segmentation_results["detected_ingredients"]
            )
            
            # Attendre que l'utilisateur ferme la fenêtre avant de continuer
            input("Appuyez sur Entrée pour passer à l'image suivante...")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {image_path.name}: {e}")
            continue

if __name__ == "__main__":
    main() 