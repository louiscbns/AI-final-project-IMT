import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
import os
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_modified_metadata(file_path):
    """
    Charge les métadonnées modifiées depuis le fichier JSON
    """
    print("\n" + "="*50)
    print("ÉTAPE 1: Chargement des métadonnées")
    print("="*50)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"✅ Données chargées avec succès: {len(data)} plats trouvés")
            return data
    except Exception as e:
        print(f"❌ Erreur lors du chargement des métadonnées : {e}")
        logger.error(f"Erreur lors du chargement des métadonnées : {e}")
        return None

def generate_text_embeddings_transformer(texts, model_name="bert-base-uncased"):
    """
    Génère des embeddings pour une liste de textes en utilisant BERT
    """
    print("\n" + "="*50)
    print("ÉTAPE 2: Génération des embeddings")
    print("="*50)
    print(f"📝 Nombre de textes à traiter: {len(texts)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Utilisation du device: {device}")
    
    try:
        print("🔄 Chargement du modèle BERT...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        print("✅ Modèle BERT chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        return None

    print("🔄 Tokenization des textes...")
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    print("✅ Tokenization terminée")

    print("🔄 Génération des embeddings...")
    with torch.no_grad():
        outputs = model(**encoded_input)
        text_features = outputs.last_hidden_state.mean(dim=1)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convertir en numpy pour l'affichage
    text_features_np = text_features.cpu().numpy()
    
    print("\n📊 Informations sur les embeddings générés:")
    print(f"   - Nombre total d'embeddings: {len(text_features_np)}")
    print(f"   - Dimension de chaque embedding: {text_features_np.shape[1]}")
    
    # Afficher les statistiques pour chaque embedding
    print("\n📈 Statistiques par embedding:")
    for i in range(len(text_features_np)):
        embedding = text_features_np[i]
        print(f"\nEmbedding {i+1}:")
        print(f"   - Moyenne: {np.mean(embedding):.4f}")
        print(f"   - Écart-type: {np.std(embedding):.4f}")
        print(f"   - Min: {np.min(embedding):.4f}")
        print(f"   - Max: {np.max(embedding):.4f}")
        print(f"   - Norme L2: {np.linalg.norm(embedding):.4f}")
        
        # Afficher les 5 premières dimensions
        print(f"   - 5 premières dimensions: {embedding[:5]}")
        
        # Afficher un extrait du texte correspondant
        print(f"   - Texte: {texts[i][:100]}...")
    
    print("\n✅ Embeddings générés avec succès")
    return text_features_np

def normalize_labels(y, output_dir='food-nutrients'):
    """
    Normalise les labels en utilisant StandardScaler
    """
    scaler = StandardScaler()
    y_normalized = scaler.fit_transform(y)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les paramètres de normalisation
    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale.npy'), scaler.scale_)
    
    return y_normalized

def prepare_data(metadata):
    """
    Prépare les données pour la génération des embeddings
    """
    print("\n" + "="*50)
    print("ÉTAPE 3: Préparation des données")
    print("="*50)
    
    X = []  # Textes pour les embeddings
    y = []  # Labels (calories, protéines, etc.)
    ids = []  # IDs des plats
    
    print("🔄 Extraction des descriptions et caractéristiques nutritionnelles...")
    total_plats = len(metadata)
    print(f"📊 Nombre total de plats à traiter: {total_plats}")
    
    for i, dish in enumerate(metadata):
        text = dish.get('description', '')
        if text:
            X.append(text)
            features = [
                dish.get('total_calories', 0),
                dish.get('total_protein', 0),
                dish.get('total_carb', 0),
                dish.get('total_fat', 0)
            ]
            y.append(features)
            ids.append(dish.get('id', f'unknown_{i}'))
            
            # Afficher les informations tous les 100 plats
            if (i + 1) % 100 == 0:
                print(f"\nProgression: {i+1}/{total_plats} plats traités ({(i+1)/total_plats*100:.1f}%)")
                print(f"Plat {i+1}:")
                print(f"Description: {text[:100]}...")
                print(f"Caractéristiques: Calories={features[0]}, Protéines={features[1]}, Glucides={features[2]}, Lipides={features[3]}")
    
    print(f"\n✅ Données préparées: {len(X)} échantillons")
    
    # Convertir en array numpy et normaliser les labels
    y_array = np.array(y)
    y_normalized = normalize_labels(y_array, output_dir='food-nutrients')
    
    print(f"📊 Statistiques des caractéristiques nutritionnelles (après normalisation):")
    for i, feature_name in enumerate(['Calories', 'Protéines', 'Glucides', 'Lipides']):
        print(f"   {feature_name}:")
        print(f"      - Moyenne: {np.mean(y_normalized[:, i]):.2f}")
        print(f"      - Écart-type: {np.std(y_normalized[:, i]):.2f}")
        print(f"      - Min: {np.min(y_normalized[:, i]):.2f}")
        print(f"      - Max: {np.max(y_normalized[:, i]):.2f}")
    
    return X, y_normalized, ids

def save_data(X_embeddings, y, output_dir='food-nutrients'):
    """
    Sauvegarde les embeddings et les labels dans un seul fichier JSON
    """
    print("\n" + "="*50)
    print("ÉTAPE 4: Sauvegarde des données")
    print("="*50)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Préparer les données pour la sauvegarde
    data_dict = {
        'embeddings': X_embeddings.tolist(),
        'labels': y.tolist(),
        'metadata': {
            'embeddings_shape': X_embeddings.shape,
            'embeddings_dtype': str(X_embeddings.dtype),
            'labels_shape': y.shape,
            'labels_dtype': str(y.dtype),
            'feature_names': ['calories', 'proteines', 'glucides', 'lipides']
        }
    }
    
    # Sauvegarder toutes les données dans un seul fichier JSON
    with open(os.path.join(output_dir, 'model_data.json'), 'w') as f:
        json.dump(data_dict, f)
    print("✅ Données sauvegardées dans 'food-nutrients/model_data.json'")

def main():
    print("\n🚀 Démarrage de la génération des embeddings")
    
    # Charger les métadonnées modifiées
    metadata = load_modified_metadata('food-nutrients/metadata_modified.json')
    if metadata is None:
        return
    
    # Préparer les données
    X, y, _ = prepare_data(metadata)
    
    # Générer les embeddings
    X_embeddings = generate_text_embeddings_transformer(X)
    
    if X_embeddings is not None:
        # Sauvegarder les données
        save_data(X_embeddings, y)
    else:
        print("❌ Échec de la génération des embeddings")

if __name__ == "__main__":
    main() 