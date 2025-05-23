import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class MultiLabelDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        """
        Args:
            json_file (string): Chemin vers le fichier JSON contenant les annotations
            image_dir (string): Répertoire contenant les images
            transform (callable, optional): Transformations à appliquer aux images
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
        
        # Extraction des ingrédients uniques à partir du fichier metadata_modified.json
        self.unique_ingredients = set()
        for food_item in self.data:
            if 'name' in food_item:
                self.unique_ingredients.add(food_item['name'])
        
        self.unique_ingredients = sorted(list(self.unique_ingredients))
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.unique_ingredients)}
        
        print(f"Nombre d'ingrédients uniques : {len(self.unique_ingredients)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        food_item = self.data[idx]
        # Utilisation de l'ID pour le chemin de l'image
        img_path = os.path.join(self.image_dir, f"{food_item['id']}.png")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image non trouvée : {img_path}")
            # Si l'image n'existe pas, on utilise une image noire
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Création du vecteur binaire des ingrédients
        label = torch.zeros(len(self.unique_ingredients), dtype=torch.float32)
        if 'name' in food_item:
            idx = self.ingredient_to_idx[food_item['name']]
            label[idx] = 1.0
            
        return image, label

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelResNet, self).__init__()
        # Utilisation de weights au lieu de pretrained
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        
        # Ajout de couches supplémentaires pour plus de stabilité
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Vérification des valeurs NaN dans les entrées
            if torch.isnan(images).any() or torch.isnan(labels).any():
                print("Warning: NaN detected in input data")
                print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
                continue
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Vérification des valeurs NaN dans les sorties
            if torch.isnan(outputs).any():
                print("Warning: NaN detected in model outputs")
                print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")
                continue
                
            loss = criterion(outputs, labels)
            
            # Vérification de la perte
            if torch.isnan(loss):
                print("Warning: NaN detected in loss")
                print(f"Labels min: {labels.min()}, max: {labels.max()}")
                continue
                
            loss.backward()
            
            # Gradient clipping pour éviter les explosions de gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
        epoch_loss = running_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Sauvegarde du meilleur modèle
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/best_model.pth')

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")
    
    json_file = 'data/food-nutrients/metadata_modified.json'
    image_dir = 'data/food-nutrients/test'
    
    # Création du dossier images s'il n'existe pas
    os.makedirs(image_dir, exist_ok=True)
    
    # Transformations pour les images avec augmentation de données
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Création du dataset
    dataset = MultiLabelDataset(json_file, image_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Initialisation du modèle
    model = MultiLabelResNet(num_classes=len(dataset.unique_ingredients))
    model = model.to(device)
    
    # Critère et optimiseur avec learning rate plus petit
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Entraînement
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Sauvegarde du modèle final
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'ingredient_mapping': dataset.ingredient_to_idx,
        'unique_ingredients': dataset.unique_ingredients
    }, 'models/multilabel_resnet18.pth')

if __name__ == '__main__':
    main() 