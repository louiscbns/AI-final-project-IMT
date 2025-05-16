import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from split_train_test import load_and_prepare_data, split_data, save_splits

# Colonnes nutritionnelles à analyser
colonnes_nutrition = ['total_calories', 'total_fat', 'total_carb', 'total_protein']

# Charger et préparer le dataset
df = load_and_prepare_data('food-nutrients/metadata.jsonl')

# Diviser les données
X_train, X_test, y_train, y_test, df_clean = split_data(df)

# Sauvegarder les ensembles
save_splits(X_train, X_test, y_train, y_test)

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
print(df_clean[colonnes_nutrition].describe())

print("\n=== TESTS DE NORMALITÉ (Shapiro-Wilk) ===")
for col in colonnes_nutrition:
    stat, p_value = stats.shapiro(df_clean[col])
    print(f"\n{col}:")
    print(f"Statistique: {stat:.3f}")
    print(f"p-value: {p_value:.3f}")

print("\n=== TAILLE DES ENSEMBLES TRAIN/TEST ===")
print(f"Train: {len(X_train)} échantillons ({len(X_train)/len(df_clean)*100:.1f}%)")
print(f"Test: {len(X_test)} échantillons ({len(X_test)/len(df_clean)*100:.1f}%)")

print("\n=== COMPARAISON DES DISTRIBUTIONS TRAIN/TEST ===")
print("\nTest de Kolmogorov-Smirnov et écarts relatifs :")
for col in colonnes_nutrition:
    ks_stat, ks_pval = stats.ks_2samp(X_train[col], X_test[col])
    mean_diff = abs(X_train[col].mean() - X_test[col].mean()) / X_train[col].mean() * 100
    std_diff = abs(X_train[col].std() - X_test[col].std()) / X_train[col].std() * 100
    print(f"\n{col}:")
    print(f"KS p-value: {ks_pval:.3f}")
    print(f"Différence moyenne: {mean_diff:.1f}%")
    print(f"Différence écart-type: {std_diff:.1f}%")

print("\n=== STATISTIQUES DES RATIOS NUTRITIONNELS ===")
print(df_clean[['ratio_proteines', 'ratio_lipides', 'ratio_glucides']].describe()) 