import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128]):
        """
        Initialise le modèle de prédiction nutritionnelle.
        
        Args:
            input_dim (int): Dimension des features d'entrée
            hidden_dims (list): Liste des dimensions des couches cachées
        """
        super().__init__()
        
        # Construction des couches du réseau
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie pour les 4 valeurs nutritionnelles
        layers.append(nn.Linear(prev_dim, 4))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass du modèle.
        
        Args:
            x (torch.Tensor): Features d'entrée
            
        Returns:
            torch.Tensor: Prédictions nutritionnelles
        """
        return self.model(x)
    
    def predict(self, features):
        """
        Prédit les valeurs nutritionnelles à partir des features.
        
        Args:
            features (torch.Tensor): Features extraites de l'image
            
        Returns:
            dict: Prédictions nutritionnelles
        """
        self.eval()
        with torch.no_grad():
            predictions = self(features)
            
        # Conversion des prédictions en dictionnaire
        nutrition_values = {
            "calories": float(predictions[0][0]),
            "proteines": float(predictions[0][1]),
            "lipides": float(predictions[0][2]),
            "glucides": float(predictions[0][3])
        }
        
        return nutrition_values
    
    def save_model(self, path):
        """
        Sauvegarde le modèle.
        
        Args:
            path (str): Chemin où sauvegarder le modèle
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Modèle sauvegardé à {path}")
    
    def load_model(self, path):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path (str): Chemin vers le modèle sauvegardé
        """
        self.load_state_dict(torch.load(path))
        logger.info(f"Modèle chargé depuis {path}") 