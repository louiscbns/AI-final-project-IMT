import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Charger le dataset depuis les fichiers locaux
def charger_dataset():
    # Charger les métadonnées
    with open('food-nutrients/metadata.jsonl', 'r') as f:
        metadata = [json.loads(line) for line in f]
    df = pd.DataFrame(metadata)
    return df

# Fonction pour l'analyse statistique descriptive
def analyse_statistique(df, colonnes):
    stats_df = pd.DataFrame()
    for col in colonnes:
        stats = {
            'Moyenne': df[col].mean(),
            'Écart-type': df[col].std(),
            'CV': df[col].std() / df[col].mean(),
            'Q1': df[col].quantile(0.25),
            'Q3': df[col].quantile(0.75),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurtosis()
        }
        stats_df[col] = pd.Series(stats)
    return stats_df

# Fonction pour le test de normalité
def test_normalite(df, colonnes):
    results = {}
    for col in colonnes:
        stat, p_value = stats.shapiro(df[col])
        results[col] = {'statistique': stat, 'p_value': p_value}
    return pd.DataFrame(results)

# Fonction pour identifier les outliers (IQR)
def identifier_outliers_mask(df, colonnes):
    mask = pd.Series([False]*len(df))
    for col in colonnes:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = mask | (df[col] < lower_bound) | (df[col] > upper_bound)
    return mask

# Fonction pour comparer les distributions train/test
def comparer_distributions(train, test, colonnes):
    results = {}
    for col in colonnes:
        ks_stat, ks_pval = stats.ks_2samp(train[col], test[col])
        mean_diff = abs(train[col].mean() - test[col].mean()) / train[col].mean() * 100
        std_diff = abs(train[col].std() - test[col].std()) / train[col].std() * 100
        results[col] = {
            'KS_stat': ks_stat,
            'KS_pval': ks_pval,
            'Diff_moyenne_%': mean_diff,
            'Diff_ecart_type_%': std_diff
        }
    return pd.DataFrame(results)

# Colonnes nutritionnelles à analyser
colonnes_nutrition = ['total_calories', 'total_fat', 'total_carb', 'total_protein']

# Charger et nettoyer le dataset
df = charger_dataset()

# Calculer les ratios nutritionnels (avant nettoyage pour éviter division par zéro)
df['ratio_proteines'] = df['total_protein'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])
df['ratio_lipides'] = df['total_fat'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])
df['ratio_glucides'] = df['total_carb'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])

# Identification et suppression des outliers sur tout le dataset
outliers_mask = identifier_outliers_mask(df, colonnes_nutrition)
df_clean = df[~outliers_mask].reset_index(drop=True)
print(f"Suppression de {outliers_mask.sum()} outliers. Taille du dataset nettoyé : {len(df_clean)}")

# Split train/test stratifié (80/20) sur le dataset nettoyé
X = df_clean[colonnes_nutrition]
y = df_clean['total_calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5))

# Créer une figure pour toutes les visualisations
plt.figure(figsize=(20, 15))

# 1. Boxplot pour total_calories
plt.subplot(2, 3, 1)
sns.boxplot(y=df_clean['total_calories'])
plt.title('Distribution des calories (nettoyé)')
plt.ylabel('total_calories')
plt.xlabel('')

stats_text_cal = "Statistiques calories:\n"
stats_text_cal += f"Moyenne: {df_clean['total_calories'].mean():.2f}\n"
stats_text_cal += f"Écart-type: {df_clean['total_calories'].std():.2f}\n"
stats_text_cal += f"Skewness: {df_clean['total_calories'].skew():.2f}\n"
stats_text_cal += f"Kurtosis: {df_clean['total_calories'].kurtosis():.2f}"
plt.text(1.1, 0.5, stats_text_cal, transform=plt.gca().transAxes, verticalalignment='center')

# 2. Boxplot pour fat, carb, protein
plt.subplot(2, 3, 2)
sns.boxplot(data=df_clean[['total_fat', 'total_carb', 'total_protein']])
plt.title('Distribution des macronutriments (nettoyé)')
plt.xticks(rotation=45)

stats_text_macro = "Statistiques macronutriments:\n"
for col in ['total_fat', 'total_carb', 'total_protein']:
    stats_text_macro += f"\n{col}:\n"
    stats_text_macro += f"Moyenne: {df_clean[col].mean():.2f}\n"
    stats_text_macro += f"Écart-type: {df_clean[col].std():.2f}\n"
    stats_text_macro += f"Skewness: {df_clean[col].skew():.2f}\n"
    stats_text_macro += f"Kurtosis: {df_clean[col].kurtosis():.2f}"
plt.text(1.1, 0.5, stats_text_macro, transform=plt.gca().transAxes, verticalalignment='center')

# 3. Matrice de corrélation
plt.subplot(2, 3, 3)
sns.heatmap(df_clean[colonnes_nutrition].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation (nettoyé)')

# 4. Distribution des calories (train vs test)
plt.subplot(2, 3, 4)
sns.kdeplot(data=y_train, label='Train')
sns.kdeplot(data=y_test, label='Test')
plt.title('Distribution des calories (Train vs Test, nettoyé)')
plt.legend()

comparaison_text = "Comparaison Train/Test:\n"
for col in colonnes_nutrition:
    ks_stat, ks_pval = stats.ks_2samp(X_train[col], X_test[col])
    mean_diff = abs(X_train[col].mean() - X_test[col].mean()) / X_train[col].mean() * 100
    std_diff = abs(X_train[col].std() - X_test[col].std()) / X_train[col].std() * 100
    comparaison_text += f"\n{col}:\n"
    comparaison_text += f"KS p-value: {ks_pval:.3f}\n"
    comparaison_text += f"Diff moyenne: {mean_diff:.1f}%\n"
    comparaison_text += f"Diff écart-type: {std_diff:.1f}%"
plt.text(1.02, 0.5, comparaison_text, transform=plt.gca().transAxes, verticalalignment='center')

# 5. Distribution des ratios nutritionnels
plt.subplot(2, 3, 5)
sns.boxplot(data=df_clean[['ratio_proteines', 'ratio_lipides', 'ratio_glucides']])
plt.title('Distribution des ratios nutritionnels (nettoyé)')
plt.xticks(rotation=45)

ratios_text = "Statistiques des ratios:\n"
for ratio in ['ratio_proteines', 'ratio_lipides', 'ratio_glucides']:
    ratios_text += f"\n{ratio}:\n"
    ratios_text += f"Moyenne: {df_clean[ratio].mean():.3f}\n"
    ratios_text += f"Écart-type: {df_clean[ratio].std():.3f}\n"
    ratios_text += f"Skewness: {df_clean[ratio].skew():.2f}\n"
    ratios_text += f"Kurtosis: {df_clean[ratio].kurtosis():.2f}"
plt.text(1.1, 0.5, ratios_text, transform=plt.gca().transAxes, verticalalignment='center')

plt.tight_layout()
plt.show()

# Afficher les résultats statistiques détaillés dans la console
print("\n=== STATISTIQUES DESCRIPTIVES ===")
stats_desc = analyse_statistique(df_clean, colonnes_nutrition)
print(stats_desc)

print("\n=== TESTS DE NORMALITÉ (Shapiro-Wilk) ===")
normalite = test_normalite(df_clean, colonnes_nutrition)
print(normalite)

print("\n=== NOMBRE D'OUTLIERS PAR VARIABLE (après nettoyage) ===")
outliers = identifier_outliers_mask(df_clean, colonnes_nutrition)
print(outliers.value_counts())

print("\n=== TAILLE DES ENSEMBLES TRAIN/TEST ===")
print(f"Train: {len(X_train)} échantillons ({len(X_train)/len(df_clean)*100:.1f}%)")
print(f"Test: {len(X_test)} échantillons ({len(X_test)/len(df_clean)*100:.1f}%)")

print("\n=== COMPARAISON DES DISTRIBUTIONS TRAIN/TEST ===")
comparaison = comparer_distributions(X_train, X_test, colonnes_nutrition)
print("\nTest de Kolmogorov-Smirnov et écarts relatifs :")
print(comparaison)
print("\nInterprétation :")
print("- KS_pval > 0.05 indique que les distributions sont similaires")
print("- Diff_moyenne_% et Diff_ecart_type_% montrent l'écart relatif en pourcentage")

print("\n=== STATISTIQUES DES RATIOS NUTRITIONNELS ===")
ratios_stats = analyse_statistique(df_clean, ['ratio_proteines', 'ratio_lipides', 'ratio_glucides'])
print(ratios_stats) 