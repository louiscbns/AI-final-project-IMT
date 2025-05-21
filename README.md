# Projet d'Analyse et Prédiction de Données Nutritionnelles

Ce projet vise à analyser et prédire les valeurs nutritionnelles des aliments en utilisant différentes approches de machine learning.

## Structure du Projet

```
projet/
├── data/                  # Données brutes
│   └── food-nutrients/    # Données nutritionnelles
├── src/                   # Code source
│   ├── data_processing/   # Scripts de traitement des données
│   ├── models/           # Implémentations des modèles et modèles sauvegardés
│   └── utils/            # Utilitaires
├── notebooks/            # Notebooks Jupyter
└── requirements.txt      # Dépendances Python
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Unix/MacOS
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Fonctionnalités

- Analyse exploratoire des données nutritionnelles
- Prédiction des calories et nutriments
- Comparaison de différents modèles (Régression Linéaire, Réseaux de Neurones)
- Visualisation des résultats

## Utilisation

1. Analyse des données :
```bash
python src/data_processing/row_data_analysis.py
```

2. Entraînement des modèles :
```bash
python src/models/best_model_to_use_determination.py
```

## Dépendances

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0 