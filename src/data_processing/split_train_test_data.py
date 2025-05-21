import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

def load_and_prepare_data(file_path):
    """
    Charge et prépare les données depuis un fichier JSONL.
    """
    # Charger les données
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convertir en DataFrame
    df = pd.DataFrame(data)
    
    # Calculer les ratios nutritionnels
    df['ratio_proteines'] = df['total_protein'] / df['total_calories']
    df['ratio_lipides'] = df['total_fat'] / df['total_calories']
    df['ratio_glucides'] = df['total_carb'] / df['total_calories']
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.
    """
    # Sélectionner les colonnes pour X et y
    X_columns = ['total_fat', 'total_carb', 'total_protein']
    y_column = 'total_calories'
    
    # Nettoyer les données (supprimer les lignes avec des valeurs manquantes)
    df_clean = df.dropna(subset=X_columns + [y_column])
    
    # Diviser les données
    X = df_clean[X_columns]
    y = df_clean[y_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, df_clean 