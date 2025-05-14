import torch
import clip
from PIL import Image
import numpy as np
import os
import ssl
import json

# Configuration du contexte SSL
ssl._create_default_https_context = ssl._create_unverified_context

def load_metadata(file_path, num_samples=10):
    """
    Charge les données depuis le fichier metadata.jsonl
    
    Args:
        file_path (str): Chemin vers le fichier metadata.jsonl
        num_samples (int): Nombre d'échantillons à charger
        
    Returns:
        list: Liste des plats avec leurs ingrédients et informations nutritionnelles
    """
    dishes_data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            dishes_data.append({
                'id': data['id'],
                'ingredients': data['ingredients'],
                'total_calories': data['total_calories'],
                'total_mass': data['total_mass'],
                'total_fat': data['total_fat'],
                'total_carb': data['total_carb'],
                'total_protein': data['total_protein']
            })
    return dishes_data

def create_ingredient_description(ingredient):
    """
    Crée une description textuelle riche pour un ingrédient
    
    Args:
        ingredient (dict): Dictionnaire contenant les informations de l'ingrédient
        
    Returns:
        str: Description textuelle de l'ingrédient
    """
    return f"{ingredient['name']} ({ingredient['grams']}g) - Calories: {ingredient['calories']:.1f}, Fat: {ingredient['fat']:.1f}g, Carbs: {ingredient['carb']:.1f}g, Protein: {ingredient['protein']:.1f}g"

def generate_text_embeddings_clip(texts):
    """
    Génère les embeddings pour une liste de textes en utilisant CLIP.
    
    Args:
        texts (list): Liste des textes (ingrédients)
        
    Returns:
        numpy.ndarray: Matrice des embeddings pour chaque texte
    """
    # Chargement du modèle CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        print("Tentative de téléchargement manuel du modèle...")
        cache_dir = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root=cache_dir)
    
    # Tokenization des textes
    text_tokens = clip.tokenize(texts).to(device)
    
    # Génération des embeddings
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()

def generate_image_embeddings_clip(images):
    """
    Génère les embeddings pour une liste d'images en utilisant CLIP.
    
    Args:
        images (list): Liste des chemins vers les images
        
    Returns:
        numpy.ndarray: Matrice des embeddings pour chaque image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        print("Tentative de téléchargement manuel du modèle...")
        cache_dir = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root=cache_dir)
    
    # Prétraitement et encodage des images
    image_features = []
    for image_path in images:
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature)
    
    return torch.cat(image_features).cpu().numpy()

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des données depuis metadata.jsonl
    metadata_path = "food-nutrients/metadata.jsonl"
    dishes_data = load_metadata(metadata_path)
    
    # Affichage des informations pour chaque plat
    for i, dish in enumerate(dishes_data, 1):
        print(f"\n{'='*80}")
        print(f"Plat {i} (ID: {dish['id']})")
        print(f"{'='*80}")
        
        print("\nIngrédients :")
        print("-" * 80)
        for ingredient in dish['ingredients']:
            print(f"• {ingredient['name']} ({ingredient['grams']}g)")
            print(f"  - Calories: {ingredient['calories']:.1f} kcal")
            print(f"  - Lipides: {ingredient['fat']:.1f}g")
            print(f"  - Glucides: {ingredient['carb']:.1f}g")
            print(f"  - Protéines: {ingredient['protein']:.1f}g")
        
        print("\nTotaux du plat :")
        print("-" * 80)
        print(f"• Masse totale: {dish['total_mass']:.1f}g")
        print(f"• Calories totales: {dish['total_calories']:.1f} kcal")
        print(f"• Lipides totaux: {dish['total_fat']:.1f}g")
        print(f"• Glucides totaux: {dish['total_carb']:.1f}g")
        print(f"• Protéines totales: {dish['total_protein']:.1f}g")
    
    # Création des descriptions enrichies pour chaque ingrédient
    all_ingredients = []
    for dish in dishes_data:
        all_ingredients.extend(dish['ingredients'])
    
    print(f"\nNombre total de plats : {len(dishes_data)}")
    print(f"Nombre total d'ingrédients : {len(all_ingredients)}")
    print("\nNombre d'ingrédients par plat :")
    for i, dish in enumerate(dishes_data, 1):
        print(f"Plat {i}: {len(dish['ingredients'])} ingrédients")
    
    enriched_descriptions = [create_ingredient_description(ing) for ing in all_ingredients]
    
    # Génération des embeddings texte
    text_embeddings = generate_text_embeddings_clip(enriched_descriptions)
    print(f"\nShape des embeddings texte : {text_embeddings.shape}")
    print("(nombre d'ingrédients, dimension des embeddings CLIP)")
    
    # Pour la partie image (à décommenter quand vous aurez des images)
    # image_paths = ["chemin/vers/image1.jpg", "chemin/vers/image2.jpg"]
    # image_embeddings = generate_image_embeddings_clip(image_paths)
    # print(f"Shape des embeddings image : {image_embeddings.shape}")
