import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
import os
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_modified_metadata(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Charge les métadonnées modifiées depuis le fichier JSON.
    
    Args:
        file_path (str): Chemin vers le fichier JSON
    
    Returns:
        Optional[List[Dict[str, Any]]]: Données chargées ou None en cas d'erreur
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

def generate_text_embeddings_transformer(texts: List[str], model_name: str = "bert-base-uncased", batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Génère des embeddings pour une liste de textes en utilisant BERT.
    
    Args:
        texts (List[str]): Liste des textes à traiter
        model_name (str): Nom du modèle BERT à utiliser
        batch_size (int): Taille des lots pour le traitement
    
    Returns:
        Optional[np.ndarray]: Embeddings générés ou None en cas d'erreur
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

    all_embeddings = []
    total_texts = len(texts)

    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"🔄 Traitement du lot {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}...")

        print("🔄 Tokenization du lot...")
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        print("✅ Tokenization du lot terminée")

        print("🔄 Génération des embeddings du lot...")
        with torch.no_grad():
            outputs = model(**encoded_input)
            batch_features = outputs.last_hidden_state.mean(dim=1)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(batch_features.cpu().numpy())
        print("✅ Embeddings du lot générés avec succès")

        # Libérer la mémoire
        del encoded_input, outputs, batch_features
        if device == "cuda":
            torch.cuda.empty_cache()

    text_features_np = np.concatenate(all_embeddings, axis=0)

    print("\n📊 Informations sur les embeddings générés:")
    print(f"   - Nombre total d'embeddings: {len(text_features_np)}")
    print(f"   - Dimension de chaque embedding: {text_features_np.shape[1]}")

    return text_features_np

def normalize_labels(y: np.ndarray, output_dir: str) -> np.ndarray:
    """
    Normalise les labels en utilisant StandardScaler.
    
    Args:
        y (np.ndarray): Labels à normaliser
        output_dir (str): Répertoire de sortie pour les paramètres de normalisation
    
    Returns:
        np.ndarray: Labels normalisés
    """
    scaler = StandardScaler()
    y_normalized = scaler.fit_transform(y)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder les paramètres de normalisation
    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale.npy'), scaler.scale_)

    return y_normalized

def prepare_data(metadata: List[Dict[str, Any]], output_dir: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Prépare les données pour la génération des embeddings.
    
    Args:
        metadata (List[Dict[str, Any]]): Métadonnées des plats
        output_dir (str): Répertoire de sortie
    
    Returns:
        Tuple[List[str], np.ndarray, List[str]]: (textes, labels normalisés, ids)
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

            if (i + 1) % 100 == 0:
                print(f"\nProgression: {i+1}/{total_plats} plats traités ({(i+1)/total_plats*100:.1f}%)")
                print(f"Plat {i+1}:")
                print(f"Description: {text[:100]}...")
                print(f"Caractéristiques: Calories={features[0]}, Protéines={features[1]}, Glucides={features[2]}, Lipides={features[3]}")

    print(f"\n✅ Données préparées: {len(X)} échantillons")

    # Convertir en array numpy et normaliser les labels
    y_array = np.array(y)
    y_normalized = normalize_labels(y_array, output_dir=output_dir)

    print(f"📊 Statistiques des caractéristiques nutritionnelles (après normalisation):")
    for i, feature_name in enumerate(['Calories', 'Protéines', 'Glucides', 'Lipides']):
        print(f"   {feature_name}:")
        print(f"      - Moyenne: {np.mean(y_normalized[:, i]):.2f}")
        print(f"      - Écart-type: {np.std(y_normalized[:, i]):.2f}")
        print(f"      - Min: {np.min(y_normalized[:, i]):.2f}")
        print(f"      - Max: {np.max(y_normalized[:, i]):.2f}")

    return X, y_normalized, ids

def save_data(X_embeddings: np.ndarray, y: np.ndarray, output_dir: str) -> None:
    """
    Sauvegarde les embeddings et les labels dans un fichier JSON.
    
    Args:
        X_embeddings (np.ndarray): Embeddings générés
        y (np.ndarray): Labels normalisés
        output_dir (str): Répertoire de sortie
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
    output_file = os.path.join(output_dir, 'model_data.json')
    with open(output_file, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"✅ Données sauvegardées dans '{output_file}'")

def main():
    """
    Fonction principale qui exécute le processus complet de génération des embeddings.
    """
    print("\n🚀 Démarrage de la génération des embeddings")
    
    output_dir = 'food-nutrients'
    
    # Charger les métadonnées modifiées
    metadata = load_modified_metadata(os.path.join(output_dir, 'metadata_modified.json'))
    if metadata is None:
        return
    
    # Préparer les données
    X, y, _ = prepare_data(metadata, output_dir)
    
    # Générer les embeddings
    X_embeddings = generate_text_embeddings_transformer(X)
    
    if X_embeddings is not None:
        # Sauvegarder les données
        save_data(X_embeddings, y, output_dir)
    else:
        print("❌ Échec de la génération des embeddings")

if __name__ == "__main__":
    main() 