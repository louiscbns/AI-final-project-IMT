import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_ingredients():
    """
    Charge le modèle entraîné et la liste des ingrédients.
    """
    # Chargement du modèle et de la liste des ingrédients
    checkpoint = torch.load("image_prediction/models/food_multilabel_resnet18.pth")
    ingredient_list = checkpoint["ingredient_list"]
    
    # Initialisation du modèle ResNet18
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(ingredient_list))
    
    # Chargement des poids entraînés
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, ingredient_list

def create_segmentation_mask(image_path, probabilities, threshold=0.5):
    """
    Crée un masque de segmentation basé sur les probabilités prédites.
    
    Args:
        image_path (str): Chemin vers l'image
        probabilities (torch.Tensor): Probabilités prédites
        threshold (float): Seuil de détection
        
    Returns:
        numpy.ndarray: Masque de segmentation
    """
    # Charger l'image originale
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Créer un masque vide
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Pour chaque ingrédient détecté
    for i in range(probabilities.shape[1]):
        if probabilities[0, i] > threshold:
            # Créer un masque pour cet ingrédient
            ingredient_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Utiliser la probabilité comme intensité
            intensity = int(probabilities[0, i] * 255)
            ingredient_mask.fill(intensity)
            
            # Ajouter au masque principal
            mask = cv2.add(mask, ingredient_mask)
    
    return mask

def visualize_prediction(image_path, predicted_ingredients, probabilities):
    """
    Visualise l'image originale, le masque de segmentation et les ingrédients prédits.
    
    Args:
        image_path (str): Chemin vers l'image
        predicted_ingredients (list): Liste des ingrédients prédits
        probabilities (torch.Tensor): Probabilités prédites
    """
    # Création de la figure
    plt.figure(figsize=(15, 7))
    
    # Affichage de l'image originale
    plt.subplot(1, 3, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title('Image Originale')
    plt.axis('off')
    
    # Création et affichage du masque de segmentation
    plt.subplot(1, 3, 2)
    mask = create_segmentation_mask(image_path, probabilities)
    plt.imshow(mask, cmap='hot')
    plt.title('Segmentation des Ingrédients')
    plt.axis('off')
    
    # Affichage des ingrédients prédits
    plt.subplot(1, 3, 3)
    plt.axis('off')
    
    # Création d'un texte formaté avec les ingrédients
    ingredients_text = "Ingrédients détectés:\n\n"
    for ingredient in predicted_ingredients:
        ingredients_text += f"• {ingredient}\n"
    
    plt.text(0.1, 0.5, ingredients_text, 
             fontsize=12, 
             verticalalignment='center',
             bbox=dict(facecolor='white', 
                      edgecolor='gray',
                      alpha=0.8,
                      boxstyle='round,pad=1'))
    
    plt.title('Prédictions')
    plt.tight_layout()
    plt.show()

def predict_ingredients(image_path, model, ingredient_list):
    """
    Prédit les ingrédients présents dans l'image.
    
    Args:
        image_path (str): Chemin vers l'image
        model: Modèle chargé
        ingredient_list (list): Liste des ingrédients
        
    Returns:
        tuple: (Liste des ingrédients détectés, Probabilités prédites)
    """
    # Prétraitement de l'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
        
    # Sélection des ingrédients avec une probabilité > 0.5
    threshold = 0.5
    predicted_indices = (probabilities > threshold).nonzero()
    
    # Conversion en liste d'ingrédients
    predicted_ingredients = []
    if predicted_indices.numel() > 0:  # Si des ingrédients ont été détectés
        # Extraire les indices de la deuxième dimension (indices des ingrédients)
        ingredient_indices = predicted_indices[:, 1]
        for idx in ingredient_indices:
            predicted_ingredients.append(ingredient_list[idx.item()])
    
    return predicted_ingredients, probabilities

def main():
    # Chargement du modèle et de la liste des ingrédients
    model, ingredient_list = load_model_and_ingredients()
    logger.info(f"Modèle chargé avec {len(ingredient_list)} ingrédients")
    
    # Exemple d'utilisation avec une image
    image_path = "data/food-nutrients/test/dish_1568665012.png"  # Image existante dans le dossier test
    try:
        predicted_ingredients, probabilities = predict_ingredients(image_path, model, ingredient_list)
        logger.info("Ingrédients détectés:")
        for ingredient in predicted_ingredients:
            logger.info(f"- {ingredient}")
            
        # Visualisation des résultats
        visualize_prediction(image_path, predicted_ingredients, probabilities)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")

if __name__ == "__main__":
    main()

# Charger le CSV
df = pd.read_csv('data/food-nutrients/labels.csv')

# Filtrer les ensembles
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# (Optionnel) Sauvegarder les splits pour vérification
train_df.to_csv('data/food-nutrients/train_labels.csv', index=False)
test_df.to_csv('data/food-nutrients/test_labels.csv', index=False) 