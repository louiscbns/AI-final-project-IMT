import torch
import numpy as np
import os
import ssl
import json
from mistralai import Mistral
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Chargement des variables d'environnement
load_dotenv()

# Configuration du contexte SSL
ssl._create_default_https_context = ssl._create_unverified_context

def load_metadata(file_path, num_samples=3):
    """
    Load data from metadata.jsonl file
    
    Args:
        file_path (str): Path to metadata.jsonl file
        num_samples (int): Number of samples to load
        
    Returns:
        list: List of dishes with their ingredients and nutritional information
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


def generate_text_embeddings_transformer(texts):
    """
    Generate embeddings for a list of texts using BERT.
    
    Args:
        texts (list): List of texts (ingredients)
        
    Returns:
        numpy.ndarray: Matrix of embeddings for each text
    """
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"  # Pre-trained English BERT model
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting manual model download...")
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    
    # Text tokenization
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
        # Use mean of last hidden layer embeddings
        text_features = outputs.last_hidden_state.mean(dim=1)
        # Normalize embeddings
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Load data from metadata.jsonl
    metadata_path = "food-nutrients/metadata.jsonl"
    dishes_data = load_metadata(metadata_path)
    
    # Display information for each dish
    for i, dish in enumerate(dishes_data, 1):
        print(f"\n{'='*80}")
        print(f"Dish {i} (ID: {dish['id']})")
        print(f"{'='*80}")
        
        print("\nIngredients:")
        print("-" * 80)
        for ingredient in dish['ingredients']:
            print(f"• {ingredient['name']} ({ingredient['grams']}g)")
            print(f"  - Calories: {ingredient['calories']:.1f} kcal")
            print(f"  - Fat: {ingredient['fat']:.1f}g")
            print(f"  - Carbs: {ingredient['carb']:.1f}g")
            print(f"  - Protein: {ingredient['protein']:.1f}g")
        
        print("\nDish totals:")
        print("-" * 80)
        print(f"• Total mass: {dish['total_mass']:.1f}g")
        print(f"• Total calories: {dish['total_calories']:.1f} kcal")
        print(f"• Total fat: {dish['total_fat']:.1f}g")
        print(f"• Total carbs: {dish['total_carb']:.1f}g")
        print(f"• Total protein: {dish['total_protein']:.1f}g")
    
    # Create enriched descriptions for each dish
    enriched_descriptions = []
    for dish in dishes_data:
        try:
            description = create_ingredient_description(dish['ingredients'])
            enriched_descriptions.append(description)
        except Exception as e:
            print(f"Error generating description: {e}")
            # Default description in case of error
            enriched_descriptions.append("Description not available - API Error")
    
    print(enriched_descriptions)
    # Generate text embeddings
    text_embeddings = generate_text_embeddings_transformer(enriched_descriptions)
    print(text_embeddings)
    print(f"\nText embeddings shape: {text_embeddings.shape}") 
    print("(number of ingredients, Transformer embedding dimension)") 