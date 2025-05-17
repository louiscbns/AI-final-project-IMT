import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset from metadata.jsonl
    
    Args:
        file_path (str): Path to metadata.jsonl file
        
    Returns:
        pd.DataFrame: Prepared dataframe with nutritional information
    """
    # Load metadata
    with open(file_path, 'r') as f:
        metadata = [json.loads(line) for line in f]
    df = pd.DataFrame(metadata)
    
    # Calculate nutritional ratios
    df['ratio_proteines'] = df['total_protein'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])
    df['ratio_lipides'] = df['total_fat'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])
    df['ratio_glucides'] = df['total_carb'] / (df['total_protein'] + df['total_fat'] + df['total_carb'])
    
    return df

def identify_outliers(df, columns):
    """
    Identify outliers using IQR method
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to check for outliers
        
    Returns:
        pd.Series: Boolean mask of outliers
    """
    mask = pd.Series([False]*len(df))
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = mask | (df[col] < lower_bound) | (df[col] > upper_bound)
    return mask

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Define nutritional columns
    nutritional_columns = ['total_calories', 'total_fat', 'total_carb', 'total_protein']
    
    # Remove outliers
    outliers_mask = identify_outliers(df, nutritional_columns)
    df_clean = df[~outliers_mask].reset_index(drop=True)
    print(f"Removed {outliers_mask.sum()} outliers. Clean dataset size: {len(df_clean)}")
    
    # Prepare features and target
    X = df_clean[nutritional_columns]
    y = df_clean['total_calories']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=pd.qcut(y, q=5)
    )
    
    return X_train, X_test, y_train, y_test, df_clean

def save_splits(X_train, X_test, y_train, y_test, output_dir='data'):
    """
    Save train and test splits to files
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        output_dir (str): Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    
    # Save test data
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"Data splits saved to {output_dir}/")

if __name__ == "__main__":
    # Load and prepare data
    metadata_path = "food-nutrients/metadata.jsonl"
    df = load_and_prepare_data(metadata_path)
    
    # Split data
    X_train, X_test, y_train, y_test, df_clean = split_data(df)
    
    # Save splits
    save_splits(X_train, X_test, y_train, y_test)
    
    # Print dataset information
    print("\n=== Dataset Information ===")
    print(f"Total samples: {len(df)}")
    print(f"Clean samples: {len(df_clean)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print feature statistics
    print("\n=== Feature Statistics ===")
    print("\nTraining set:")
    print(X_train.describe())
    print("\nTest set:")
    print(X_test.describe()) 