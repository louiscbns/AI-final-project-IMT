import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

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
    Dénormalise les prédictions en utilisant les statistiques calculées à partir des données
    """
    try:
        data_path = Path(__file__).parent.parent.parent / 'data' / data_dir / 'embeddings.json'
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Calculer les statistiques à partir des données
        labels = np.array(data['labels'])
        stats = {}
        for i, feature in enumerate(['calories', 'proteines', 'glucides', 'lipides']):
            stats[feature] = {
                'mean': np.mean(labels[:, i]),
                'std': np.std(labels[:, i])
            }
        
        y_denormalized = y_normalized.copy()
        
        for i, feature in enumerate(['calories', 'proteines', 'glucides', 'lipides']):
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            y_denormalized[:, i] = y_denormalized[:, i] * std + mean
            
        return y_denormalized
    except Exception as e:
        logging.error(f"Erreur lors de la dénormalisation : {str(e)}")
        raise

def train_and_evaluate_models(X_embeddings, y):
    """
    Entraîne et évalue plusieurs modèles de régression pour prédire les 4 valeurs nutritionnelles
    """
    print("\n" + "="*50)
    print("ÉTAPE 2: Entraînement et évaluation des modèles")
    print("="*50)
    
    print("🔄 Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42
    )
    print(f"✅ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
    
    # Définition des modèles à tester
    models = {
        'Réseau de neurones (100,50)': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=2000,
            learning_rate_init=0.001,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-4,
            activation='relu',
            solver='adam',
            batch_size=32,
            alpha=0.001,
            random_state=42
        ),
        'Réseau de neurones (50,25)': MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=2000,
            learning_rate_init=0.001,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-4,
            activation='relu',
            solver='adam',
            batch_size=32,
            alpha=0.001,
            random_state=42
        ),
        'Réseau de neurones (200)': MLPRegressor(
            hidden_layer_sizes=(200,),
            max_iter=2000,
            learning_rate_init=0.001,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-4,
            activation='relu',
            solver='adam',
            batch_size=32,
            alpha=0.001,
            random_state=42
        ),
        'Réseau de neurones (100)': MLPRegressor(
            hidden_layer_sizes=(100,),
            max_iter=2000,
            learning_rate_init=0.001,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-4,
            activation='relu',
            solver='adam',
            batch_size=32,
            alpha=0.001,
            random_state=42
        ),
        'Réseau de neurones (50)': MLPRegressor(
            hidden_layer_sizes=(50,),
            max_iter=2000,
            learning_rate_init=0.001,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            tol=1e-4,
            activation='relu',
            solver='adam',
            batch_size=32,
            alpha=0.001,
            random_state=42
        )
    }
    
    results = {}
    feature_names = ['calories', 'proteines', 'glucides', 'lipides']
    
    for model_name, model in models.items():
        print(f"\n📊 Entraînement du modèle {model_name}...")
        model.fit(X_train, y_train)
        
        print("🔄 Prédictions...")
        y_pred = model.predict(X_test)
        
        # Dénormaliser les données pour l'évaluation
        y_test_denorm = denormalize_predictions(y_test)
        y_pred_denorm = denormalize_predictions(y_pred)
        
        # Calcul des métriques pour chaque caractéristique
        model_metrics = {}
        for i, feature_name in enumerate(feature_names):
            mse = mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])
            mae = mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i])
            r2 = r2_score(y_test_denorm[:, i], y_pred_denorm[:, i])
            
            model_metrics[feature_name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"\n✅ {feature_name.upper()} pour {model_name}:")
            print(f"   - Erreur quadratique moyenne (MSE): {mse:.2f}")
            print(f"   - Erreur absolue moyenne (MAE): {mae:.2f}")
            print(f"   - Score R²: {r2:.2f}")
        
        results[model_name] = {
            'model': model,
            'metrics': model_metrics,
            'predictions': y_pred
        }
    
    # Créer un graphique comparatif des performances
    plt.figure(figsize=(15, 20))
    
    # Graphique des scores R²
    plt.subplot(5, 1, 1)
    r2_scores = pd.DataFrame({
        model: [results[model]['metrics'][feature]['R2'] for feature in feature_names]
        for model in models.keys()
    }, index=feature_names)
    r2_scores.plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des scores R² par modèle et par caractéristique')
    plt.ylabel('Score R²')
    plt.xticks(rotation=45)
    
    # Graphique des MSE pour les calories
    plt.subplot(5, 1, 2)
    mse_calories = pd.DataFrame({
        model: [results[model]['metrics']['calories']['MSE']]
        for model in models.keys()
    }, index=['Calories'])
    mse_calories.plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des MSE pour les calories par modèle')
    plt.ylabel('MSE Calories')
    plt.xticks(rotation=45)
    
    # Graphique des MSE pour les autres nutriments
    plt.subplot(5, 1, 3)
    mse_others = pd.DataFrame({
        model: [
            results[model]['metrics']['proteines']['MSE'],
            results[model]['metrics']['glucides']['MSE'],
            results[model]['metrics']['lipides']['MSE']
        ]
        for model in models.keys()
    }, index=['Protéines', 'Glucides', 'Lipides'])
    mse_others.plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des MSE pour les autres nutriments par modèle')
    plt.ylabel('MSE Autres nutriments')
    plt.xticks(rotation=45)
    
    # Graphique des MAE pour les calories
    plt.subplot(5, 1, 4)
    mae_calories = pd.DataFrame({
        model: [results[model]['metrics']['calories']['MAE']]
        for model in models.keys()
    }, index=['Calories'])
    mae_calories.plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des MAE pour les calories par modèle')
    plt.ylabel('MAE Calories')
    plt.xticks(rotation=45)
    
    # Graphique des MAE pour les autres nutriments
    plt.subplot(5, 1, 5)
    mae_others = pd.DataFrame({
        model: [
            results[model]['metrics']['proteines']['MAE'],
            results[model]['metrics']['glucides']['MAE'],
            results[model]['metrics']['lipides']['MAE']
        ]
        for model in models.keys()
    }, index=['Protéines', 'Glucides', 'Lipides'])
    mae_others.plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des MAE pour les autres nutriments par modèle')
    plt.ylabel('MAE Autres nutriments')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Retourner le meilleur modèle (celui avec le meilleur R² moyen)
    best_model_name = max(results.keys(), 
                         key=lambda x: np.mean([results[x]['metrics'][f]['R2'] for f in feature_names]))
    print(f"\n🏆 Meilleur modèle: {best_model_name}")
    
    return results[best_model_name]['model'], results[best_model_name]['predictions']

def main():
    print("\n🚀 Démarrage du processus d'entraînement")
    
    # Charger les données
    X_embeddings, y = load_data()
    if X_embeddings is None or y is None:
        return
    
    # Entraîner et évaluer les modèles
    best_model, y_pred = train_and_evaluate_models(X_embeddings, y)
    
    # Sauvegarder le meilleur modèle
    print("\n" + "="*50)
    print("ÉTAPE 3: Sauvegarde du modèle")
    print("="*50)
    import joblib
    joblib.dump(best_model, 'model.joblib')
    print("✅ Modèle sauvegardé dans model.joblib")

if __name__ == "__main__":
    main() 