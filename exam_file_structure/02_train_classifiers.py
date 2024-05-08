import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, Optional
import joblib

FEATURES_PATH = "features/features.csv" # Path for features data from 01_process_images
RESULTS_PATH = "features/training/results.csv" # Path to save training results to

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data by selecting specific features.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    features = ['hue', 'saturation', 'value', 'hsv_uniformity', 'compactness_score', 'Vertical Asymmetry_mean', 'Horizontal Asymmetry_mean']
    target = 'cancer'
    df = df[features + [target]]
    return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop('cancer', axis=1)
    y = df['cancer']
    return train_test_split(X, y, test_size=0.20, random_state=42)

def train(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Train models.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
        save_path (Optional[str]): Path to save trained models (default=None).
    
    Returns:
        pd.DataFrame: DataFrame containing grid search results.
    """
    param_grid = {
        'n_estimators': [*range(1, 17), 50, 100, 1000],
        'max_depth': range(1, 21),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = ParameterGrid(param_grid)
    results = pd.DataFrame()

    for params in grid:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        # print(f"Params: {params}, Accuracy: {accuracy:.4f}")
        params['accuracy'] = accuracy
        results = pd.concat([results, pd.DataFrame([params])], ignore_index=True)
        
        if save_path is not None:
            # Save model
            model_save_path = f"{save_path}model_{params}.pkl"
            joblib.dump(model, model_save_path)
    
    return results

def main(features_path: str, models_path: Optional[str] = None, results_path: Optional[str] = None) -> None:
    """
    Load data, preprocess it, split it into training and testing sets, train models, and save results.

    Parameters:
        features_path (str): The path to the CSV file containing the features data.
        models_path (Optional[str]): The directory path where trained models will be saved (default=None).
        results_path (Optional[str]): The path where the results DataFrame will be saved as a CSV file (default=None).
    """
    data = load_data(features_path)
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    
    results = train(X_train, y_train, X_test, y_test, save_path=models_path)
    
    if results_path is not None:
        # Save the results DataFrame to a CSV file
        results.to_csv(results_path, index=False)
        
if __name__ == "__main__":
    main(FEATURES_PATH, results_path=RESULTS_PATH)
