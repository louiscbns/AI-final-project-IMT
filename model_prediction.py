import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import os
from pathlib import Path
import json

def generate_text_embedding(text, model_name="bert-base-uncased"):
    """
    Génère l'embedding pour un texte en utilisant BERT
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        print("Tentative de téléchargement manuel du modèle...")
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    
    # Traiter le texte comme un lot de taille 1
    texts = [text]
    
    # Tokenization du texte
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Génération de l'embedding
    with torch.no_grad():
        outputs = model(**encoded_input)
        text_features = outputs.last_hidden_state.mean(dim=1)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Libérer la mémoire
    del encoded_input, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return text_features.cpu().numpy()

def denormalize_predictions(y_normalized, data_dir='food-nutrients'):
    """
    Dénormalise les prédictions en utilisant les paramètres de normalisation sauvegardés
    """
    try:
        data_path = Path(__file__).parent / 'data' / data_dir
        
        # Charger les paramètres de normalisation
        mean = np.load(os.path.join(data_path, 'scaler_mean.npy'))
        scale = np.load(os.path.join(data_path, 'scaler_scale.npy'))
        
        # Dénormaliser les prédictions
        y_denormalized = y_normalized * scale + mean
            
        return y_denormalized
    except Exception as e:
        print(f"Erreur lors de la dénormalisation : {str(e)}")
        raise

def predict_nutritional_values(description):
    """
    Prédit les valeurs nutritionnelles à partir d'une description de plat
    """
    # Charger le modèle
    model_path = Path(__file__).parent / 'src' / 'models' / 'neural_network_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Le modèle n'a pas été trouvé à l'emplacement : {model_path}")
    
    model = joblib.load(model_path)
    
    # Afficher les informations sur le modèle
    print("\nInformations sur le modèle utilisé :")
    print("=" * 50)
    print(f"Type de modèle : {type(model).__name__}")
    print(f"Architecture : {model.hidden_layer_sizes}")
    print(f"Fonction d'activation : {model.activation}")
    print(f"Solveur utilisé : {model.solver}")
    print(f"Taux d'apprentissage initial : {model.learning_rate_init}")
    print(f"Nombre maximum d'itérations : {model.max_iter}")
    print(f"Early stopping : {'Activé' if model.early_stopping else 'Désactivé'}")
    if model.early_stopping:
        print(f"Fraction de validation : {model.validation_fraction}")
    print("=" * 50)
    
    # Générer l'embedding pour la description
    embedding = generate_text_embedding(description)
    
    # Faire la prédiction
    y_pred_normalized = model.predict(embedding)
    
    # Dénormaliser les prédictions
    y_pred = denormalize_predictions(y_pred_normalized)
    
    # Retourner les résultats
    feature_names = ['calories', 'proteins', 'carbs', 'fats']
    results = {}
    for i, feature_name in enumerate(feature_names):
        results[feature_name] = float(y_pred[0, i])
    
    return results

if __name__ == "__main__":
    # Exemple d'utilisation
    description = input("Entrez la description du plat : ")
    results = predict_nutritional_values(description)
    
    print("\nPredicted nutritional values:")
    print("=" * 30)
    for feature, value in results.items():
        print(f"{feature.capitalize()}: {value:.1f}") 