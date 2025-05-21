import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Charge les données depuis le fichier embeddings.json"""
    try:
        data_path = Path(__file__).parent.parent.parent / 'data' / 'food-nutrients' / 'embeddings.json'
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convertir les listes en arrays numpy
        X_embeddings = np.array(data['embeddings'])
        y = np.array(data['labels'])
        
        # Afficher les informations sur les données
        logging.info(f"Données chargées avec succès:")
        logging.info(f"   - Nombre d'échantillons: {len(X_embeddings)}")
        logging.info(f"   - Dimension des embeddings: {X_embeddings.shape[1]}")
        logging.info(f"   - Nombre de caractéristiques: {y.shape[1]}")
        logging.info(f"   - Caractéristiques: {data['metadata']['feature_names']}")
        
        return X_embeddings, y
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {str(e)}")
        raise

def denormalize_predictions(y_normalized, data_dir='food-nutrients'):
    """
    Dénormalise les prédictions en utilisant les paramètres sauvegardés lors de la normalisation
    """
    try:
        base_path = Path(__file__).parent.parent.parent / 'data' / data_dir
        mean = np.load(base_path / 'scaler_mean.npy')
        scale = np.load(base_path / 'scaler_scale.npy')
        y_denormalized = y_normalized.copy()
        y_denormalized = y_denormalized * scale + mean
        return y_denormalized
    except Exception as e:
        logging.error(f"Erreur lors de la dénormalisation : {str(e)}")
        raise

def train_and_evaluate_model(X_embeddings, y):
    """
    Entraîne et évalue un modèle de régression linéaire
    """
    print("\n" + "="*50)
    print("ÉTAPE 2: Entraînement et évaluation du modèle")
    print("="*50)
    
    print("🔄 Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42
    )
    print(f"✅ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
    
    # Créer et entraîner le modèle
    print("\n📊 Entraînement du modèle de régression linéaire...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Faire des prédictions
    print("🔄 Prédictions...")
    y_pred = model.predict(X_test)
    
    # Dénormaliser les données pour l'évaluation
    y_test_denorm = denormalize_predictions(y_test)
    y_pred_denorm = denormalize_predictions(y_pred)
    
    # Créer un DataFrame pour la matrice de corrélation
    feature_names = ['calories', 'proteines', 'glucides', 'lipides']
    df_test = pd.DataFrame(y_test_denorm, columns=feature_names)
    
    # Créer la figure pour les graphiques
    plt.figure(figsize=(20, 15))
    
    # Graphiques de comparaison avec métriques
    for i, feature_name in enumerate(feature_names):
        plt.subplot(2, 3, i+1)
        plt.scatter(y_test_denorm[:, i], y_pred_denorm[:, i], alpha=0.5)
        
        # Ligne idéale (y=x)
        plt.plot([y_test_denorm[:, i].min(), y_test_denorm[:, i].max()],
                [y_test_denorm[:, i].min(), y_test_denorm[:, i].max()],
                'r--', lw=2, label='Ligne idéale')
        
        # Courbe de régression réelle
        z = np.polyfit(y_test_denorm[:, i], y_pred_denorm[:, i], 1)
        p = np.poly1d(z)
        x_range = np.linspace(y_test_denorm[:, i].min(), y_test_denorm[:, i].max(), 100)
        plt.plot(x_range, p(x_range), 'b-', lw=2, label='Régression réelle')
        
        # Calculer les métriques
        mse = mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])
        mae = mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i])
        r2 = r2_score(y_test_denorm[:, i], y_pred_denorm[:, i])
        corr = np.corrcoef(y_test_denorm[:, i], y_pred_denorm[:, i])[0,1]
        
        # Ajouter les métriques au graphique
        plt.title(f'Régression linéaire - {feature_name}\n'
                 f'R² = {r2:.3f}, Corr = {corr:.3f}\n'
                 f'MSE = {mse:.2f}, MAE = {mae:.2f}')
        plt.xlabel(f'Valeurs réelles ({feature_name})')
        plt.ylabel(f'Prédictions ({feature_name})')
        plt.legend()
    
    # Matrice de corrélation
    plt.subplot(2, 3, 5)
    correlation_matrix = df_test.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation des nutriments')
    
    plt.tight_layout()
    plt.show()
    
    return model, y_pred

def main():
    print("\n🚀 Démarrage du processus d'entraînement")
    
    # Charger les données
    X_embeddings, y = load_data()
    
    # Entraîner et évaluer le modèle
    model, y_pred = train_and_evaluate_model(X_embeddings, y)
    
    # Sauvegarder le modèle
    print("\n" + "="*50)
    print("ÉTAPE 3: Sauvegarde du modèle")
    print("="*50)
    import joblib
    model_path = Path(__file__).parent / 'linear_regression_model.joblib'
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé dans {model_path}")

if __name__ == "__main__":
    main() 