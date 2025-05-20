import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import os
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir='food-nutrients'):
    """
    Charge les embeddings et les labels depuis le fichier JSON
    """
    print("\n" + "="*50)
    print("ÉTAPE 1: Chargement des données")
    print("="*50)
    
    try:
        # Charger toutes les données depuis le fichier JSON
        with open(os.path.join(data_dir, 'model_data.json'), 'r') as f:
            data = json.load(f)
            
            # Convertir les listes en arrays numpy
            X_embeddings = np.array(data['embeddings'])
            y = np.array(data['labels'])
            
            # Afficher les informations sur les données
        print(f"✅ Données chargées avec succès:")
        print(f"   - Nombre d'échantillons: {len(X_embeddings)}")
        print(f"   - Dimension des embeddings: {X_embeddings.shape[1]}")
        print(f"   - Nombre de caractéristiques: {y.shape[1]}")
        print(f"   - Caractéristiques: {data['metadata']['feature_names']}")
            
        return X_embeddings, y
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        logger.error(f"Erreur lors du chargement des données : {e}")
        return None, None

def denormalize_predictions(y_normalized, data_dir='food-nutrients'):
    """
    Dénormalise les prédictions en utilisant les paramètres sauvegardés
    """
    mean = np.load(os.path.join(data_dir, 'scaler_mean.npy'))
    scale = np.load(os.path.join(data_dir, 'scaler_scale.npy'))
    return y_normalized * scale + mean

def plot_results(y_test, y_pred, feature_names):
    """
    Crée des visualisations pour analyser les résultats
    """
    print("\n" + "="*50)
    print("ÉTAPE 4: Visualisation des résultats")
    print("="*50)
    
    # Dénormaliser les données pour l'affichage
    y_test_denorm = denormalize_predictions(y_test)
    y_pred_denorm = denormalize_predictions(y_pred)
    
    # Créer une figure avec plusieurs sous-graphiques
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Graphiques de dispersion pour chaque caractéristique (2x2 en haut à gauche)
    for i, feature_name in enumerate(feature_names):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        plt.scatter(y_test_denorm[:, i], y_pred_denorm[:, i], alpha=0.6)
        ax.plot([y_test_denorm[:, i].min(), y_test_denorm[:, i].max()], 
                [y_test_denorm[:, i].min(), y_test_denorm[:, i].max()], 
                'r--', lw=2)
        
        # Calculer le coefficient de corrélation
        correlation = np.corrcoef(y_test_denorm[:, i], y_pred_denorm[:, i])[0, 1]
        r2 = r2_score(y_test_denorm[:, i], y_pred_denorm[:, i])
        
        # Ajouter le coefficient de corrélation et R² sur le graphique
        ax.text(0.05, 0.95, 
                f'Corrélation: {correlation:.3f}\nR²: {r2:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Valeurs réelles ({feature_name})')
        ax.set_ylabel(f'Prédictions ({feature_name})')
        ax.set_title(f'Prédictions vs Valeurs réelles - {feature_name} (n={len(y_test)})')
    
    # 2. Boîte à moustaches des erreurs pour les calories (en haut à droite)
    ax = fig.add_subplot(gs[0, 2])
    errors = y_pred_denorm - y_test_denorm
    df_errors_calories = pd.DataFrame(errors[:, 0], columns=['calories'])
    df_errors_calories.boxplot(ax=ax)
    ax.set_title('Distribution des erreurs pour les calories')
    ax.set_ylabel('Erreur (Prédiction - Réelle)')
    
    # 3. Boîte à moustaches des erreurs pour les autres nutriments (au milieu à droite)
    ax = fig.add_subplot(gs[1, 2])
    df_errors_autres = pd.DataFrame(errors[:, 1:], columns=feature_names[1:])
    df_errors_autres.boxplot(ax=ax)
    ax.set_title('Distribution des erreurs pour les autres nutriments')
    ax.set_ylabel('Erreur (Prédiction - Réelle)')
    plt.xticks(rotation=45)
    
    # 4. Matrice de corrélation des erreurs (en bas à droite)
    ax = fig.add_subplot(gs[2, 2])
    corr_matrix = np.corrcoef(errors.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm')
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticklabels(feature_names)
    
    # Ajouter les valeurs de corrélation
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center')
    
    plt.colorbar(im, ax=ax)
    ax.set_title('Corrélation des erreurs entre caractéristiques')
    
    plt.tight_layout()
    print("✅ Affichage des graphiques...")
    plt.show()
    plt.close()

def train_and_evaluate_model(X_embeddings, y):
    """
    Entraîne et évalue un seul modèle de régression multiple pour prédire les 4 valeurs nutritionnelles
    """
    print("\n" + "="*50)
    print("ÉTAPE 2: Entraînement et évaluation du modèle")
    print("="*50)
    
    print("🔄 Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42
    )
    print(f"✅ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
    
    print("\n📊 Entraînement du modèle de régression multiple...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("🔄 Prédictions...")
    y_pred = model.predict(X_test)
    
    # Calcul des métriques pour chaque caractéristique
    feature_names = ['calories', 'proteines', 'glucides', 'lipides']
    print("\n📈 Métriques pour chaque caractéristique:")
    
    # Dénormaliser les données pour l'évaluation
    y_test_denorm = denormalize_predictions(y_test)
    y_pred_denorm = denormalize_predictions(y_pred)
    
    for i, feature_name in enumerate(feature_names):
        mse = mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])
        r2 = r2_score(y_test_denorm[:, i], y_pred_denorm[:, i])
        
        print(f"\n✅ {feature_name.upper()}:")
        print(f"   - Erreur quadratique moyenne (MSE): {mse:.2f}")
        print(f"   - Score R²: {r2:.2f}")
        
        # Afficher les prédictions vs valeurs réelles pour chaque échantillon
        print(f"\n   Comparaison prédictions vs valeurs réelles pour {feature_name}:")
        for j in range(min(5, len(y_test))):  # Afficher seulement les 5 premiers échantillons
            print(f"   Échantillon {j+1}:")
            print(f"      Valeur réelle: {y_test_denorm[j, i]:.2f}")
            print(f"      Prédiction: {y_pred_denorm[j, i]:.2f}")
    
    # Afficher les coefficients du modèle
    print("\n📊 Coefficients du modèle:")
    for i, feature_name in enumerate(feature_names):
        print(f"\nCoefficients pour {feature_name}:")
        # Afficher les 5 premiers coefficients les plus importants
        coef_indices = np.argsort(np.abs(model.coef_[i]))[-5:]
        for idx in coef_indices:
            print(f"   Embedding {idx}: {model.coef_[i][idx]:.4f}")
    
    # Créer les visualisations
    plot_results(y_test, y_pred, feature_names)
    
    return model, y_pred

def main():
    print("\n🚀 Démarrage du processus d'entraînement")
    
    # Charger les données
    X_embeddings, y = load_data()
    if X_embeddings is None or y is None:
        return
    
    # Entraîner et évaluer le modèle
    model, y_pred = train_and_evaluate_model(X_embeddings, y)
    
    # Sauvegarder le modèle
    print("\n" + "="*50)
    print("ÉTAPE 3: Sauvegarde du modèle")
    print("="*50)
    import joblib
    joblib.dump(model, 'model.joblib')
    print("✅ Modèle sauvegardé dans model.joblib")

if __name__ == "__main__":
    main() 