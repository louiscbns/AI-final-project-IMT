import torch
import torchvision
from PIL import Image
import numpy as np
import logging
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodSegmentation:
    def __init__(self):
        """
        Initialise le système de segmentation des aliments.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilisation du device pour la segmentation: {self.device}")
        
        # Utilisation de DeepLabV3 avec ResNet-101 comme backbone
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Définition des classes pour la segmentation alimentaire
        self.food_classes = {
            0: "background",
            1: "plat_principal",
            2: "accompagnement",
            3: "sauce",
            4: "garniture"
        }
        
    def preprocess_for_segmentation(self, image):
        """
        Prétraite l'image pour la segmentation.
        
        Args:
            image (PIL.Image): Image à segmenter
            
        Returns:
            torch.Tensor: Image prétraitée
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((520, 520)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def postprocess_mask(self, output, original_size):
        """
        Post-traite le masque de segmentation.
        
        Args:
            output (torch.Tensor): Sortie du modèle
            original_size (tuple): Taille originale de l'image (height, width)
            
        Returns:
            numpy.ndarray: Masque de segmentation post-traité
        """
        # Redimensionnement à la taille originale
        output = F.interpolate(
            output,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Conversion en masque
        mask = output.argmax(1).squeeze().cpu().numpy()
        return mask
    
    def segment_image(self, image_path):
        """
        Segmente l'image pour identifier les différents aliments.
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            dict: Résultats de la segmentation
        """
        try:
            # Chargement de l'image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size[::-1]  # (height, width)
            
            # Prétraitement
            input_tensor = self.preprocess_for_segmentation(image)
            
            # Prédiction
            with torch.no_grad():
                output = self.model(input_tensor)['out']
                mask = self.postprocess_mask(output, original_size)
            
            # Identification des ingrédients
            detected_ingredients = self.identify_ingredients(mask)
            
            segmentation_results = {
                "segmentation_mask": mask,
                "detected_ingredients": detected_ingredients
            }
            
            return segmentation_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la segmentation: {e}")
            raise
    
    def identify_ingredients(self, segmentation_mask):
        """
        Identifie les ingrédients à partir du masque de segmentation.
        
        Args:
            segmentation_mask (numpy.ndarray): Masque de segmentation
            
        Returns:
            list: Liste des ingrédients détectés
        """
        detected_classes = np.unique(segmentation_mask)
        ingredients = []
        
        for class_id in detected_classes:
            if class_id in self.food_classes:
                ingredients.append(self.food_classes[class_id])
        
        return ingredients 