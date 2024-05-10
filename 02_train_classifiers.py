import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, Optional, Dict, List, Any
import joblib
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for specifying paths for features data, results, and optionally model saving.
    If a results file already exists at the specified path, it will be loaded instead of re-running the training process.

    Returns:
        argparse.Namespace: The namespace object containing all arguments.
    """
    parser = argparse.ArgumentParser(description="Train models and evaluate performance")
    parser.add_argument('--features_path', type=str, default='features/features.csv', help='Path to the CSV file containing features data.')
    parser.add_argument('--results_path', type=str, default='training/results.csv', help='Path where training results will be saved or loaded from if file already exists.')
    parser.add_argument('--graphs_path', type=str, default='graphs/', help='Path where graphs will be saved to.')
    parser.add_argument('--models_path', type=str, default=None, help='Directory path to save trained models. If not specified, models are not saved.')
    parser.add_argument('--final_model_path', type=str, default='training/final_model', help='Path where final model will be saved to.')
    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file. This function is used to read the data into a DataFrame which
    allows for further manipulation and analysis.
    
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
    features = ['hue', 'saturation', 'value', 'hsv_uniformity', 'compactness_score', 'Vertical Asymmetry_mean',
                'Horizontal Asymmetry_mean']
    target = 'cancer'
    df = df[features + [target]]
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
   Split data into training and testing sets. This function ensures that the model is tested on
    unseen data, evaluating the model's performance effectively.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop('cancer', axis=1)
    y = df['cancer']
    return train_test_split(X, y, test_size=0.20, random_state=42)


def train(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
          save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Train multiple RandomForest models using a grid of parameters to find the best performing model.
    Uses the ParameterGrid to create combinations of parameters, which are then applied to training the model.
    
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
        'n_estimators': [*range(1, 17), 50, 100, 1000, 2500, 5000],
        'max_depth': range(1, 21),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = ParameterGrid(param_grid)  # Creates a grid of parameter combinations from specified options.
    results = pd.DataFrame()

    if save_path is not None:
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for params in grid:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        # logging.info(f"Params: {params}, Accuracy: {accuracy:.4f}")
        params['accuracy'] = accuracy
        results = pd.concat([results, pd.DataFrame([params])], ignore_index=True)

        if save_path is not None:
            # Save model
            model_save_path = f"{save_path}model_{params}"
            joblib.dump(model, model_save_path)

    return results


def save_or_show_plot(output_file: Optional[str] = None) -> None:
    """
    Saves the plot to a PDF file if an output file path is provided, or displays the plot on screen if no file path is specified.

    Parameters:
        output_file (Optional[str]): The file path where the plot will be saved as a PDF. If None, the plot will be displayed using plt.show().

    Returns:
        None
    """
    if output_file is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the figure as a PDF
        plt.savefig(output_file)
        plt.close()  # Close the plot to free up memory
    else:
        # Display the plot
        plt.show()


def plot_heatmaps(results: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """
    Generates a series of heatmaps from a DataFrame containing model performance results, specifically
    plotting the accuracy of models across various configurations of 'max_depth' and 'n_estimators', grouped
    by combinations of 'min_samples_split', 'min_samples_leaf', and 'max_features'.

    Each heatmap corresponds to a specific combination of 'min_samples_split', 'min_samples_leaf',
    and 'max_features', showing how the accuracy varies with 'max_depth' and 'n_estimators'.

    Parameters:
        results (pd.DataFrame): A DataFrame containing the columns 'min_samples_split', 'min_samples_leaf',
                                'max_features', 'max_depth', 'n_estimators', and 'accuracy' that records
                                the accuracy of various RandomForest configurations.
        output_file (Optional[str]): The filename where the plot PDF will be saved. Defaults to 'features/training/graphs/heatmaps.pdf'.

    Returns:
        None: This function does not return any value but saves the heatmaps to a PDF file and displays them.
    """
    # Filter unique parameters for grouping
    unique_splits = results['min_samples_split'].unique()
    unique_leaves = results['min_samples_leaf'].unique()
    unique_features = results['max_features'].unique()

    # Prepare subplot grid layout
    num_plots = len(unique_splits) * len(unique_leaves) * len(unique_features)
    num_cols = 2  # Define number of columns in grid layout
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate needed rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Ensure axes is a 2D array for easier indexing
    if num_rows == 1:
        axes = np.array([axes])

    idx = 0
    for split in unique_splits:
        for leaf in unique_leaves:
            for feature in unique_features:
                ax = axes[idx // num_cols, idx % num_cols]
                # Filter results for specific parameters
                subset = results[(results['min_samples_split'] == split) &
                                 (results['min_samples_leaf'] == leaf) &
                                 (results['max_features'] == feature)]
                # Pivot table for heatmap
                pivot_table = subset.pivot_table(values='accuracy', index='max_depth', columns='n_estimators')
                sns.heatmap(pivot_table, fmt=".2f", cmap='viridis', ax=ax)
                ax.set_title(f'Split: {split}, Leaf: {leaf}, Feature: {feature}')
                ax.set_xlabel('Number of Estimators')
                ax.set_ylabel('Max Depth')
                idx += 1

    # Adjust layout
    plt.tight_layout()

    # Output plot
    save_or_show_plot(output_file)


def plot_parameter_effects(results: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """
    Create box-and-whisker plots for each parameter against model accuracy, coloring each box uniquely.

    Parameters:
        results (pd.DataFrame): DataFrame containing model parameters and accuracies.
        output_file (Optional[str]): The path where the plot will be saved as a PDF file.

    Returns:
        None: This function does not return any value but saves the boxplots to a PDF file and displays them.
    """
    parameters = [col for col in results.columns if col != 'accuracy']
    n_plots = len(parameters)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 5 * n_plots))

    if n_plots == 1:
        axes = [axes]  # Make it iterable if there is only one plot

    # Define a color palette with as many colors as there are unique values in each parameter
    for param, ax in zip(parameters, axes):
        unique_values = results[param].nunique()
        palette = sns.color_palette("hsv", unique_values)  # Create a color palette with unique hues

        sns.boxplot(data=results, x=param, y='accuracy', ax=ax, palette=palette)
        ax.set_title(f'Model Accuracy Distribution by {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Accuracy')
        ax.grid(True)

    plt.tight_layout()  # Adjust layout
    save_or_show_plot(output_file)  # Output plot


def plot_accuracy_vs_estimators(results: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """
    Plots a scatter plot of model accuracy against the logarithm of the number of estimators ('n_estimators') used in the models,
    including a regression line to indicate the trend. The x-axis (number of estimators) is displayed on a logarithmic scale.

    Parameters:
        results (pd.DataFrame): DataFrame containing at least 'n_estimators' and 'accuracy'.
        output_file (Optional[str]): The path where the plot will be saved as a PDF file.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    # Create a copy of the DataFrame to ensure the original is not modified
    results_copy = results.copy()
    # Adjusting the x-axis to be logarithmic by adding a new column to the copy
    results_copy['log_n_estimators'] = np.log10(results_copy['n_estimators'])

    # Create scatter plot with regression line on the copied data
    sns.regplot(x='log_n_estimators', y='accuracy', data=results_copy, scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'})
    plt.title('Model Accuracy vs. Log of Number of Estimators')
    plt.xlabel('Log of Number of Estimators (log10)')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    save_or_show_plot(output_file)


def select_best_parameters(results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Selects the best parameters for models based on different statistics of the 'accuracy' column:
    highest median, mean, first quartile (Q1), third quartile (Q3), and maximum accuracy.

    Parameters:
        results (pd.DataFrame): A DataFrame containing model parameters and their corresponding 'accuracy'.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a statistic ('median', 'mean', 'Q1', 'Q3', 'max')
                                     and the value is a dictionary of the parameters that maximize that statistic.
    """
    best_params = {
        'median': {},
        'mean': {},
        'Q1': {},
        'Q3': {},
        'max': {}
    }

    # Exclude 'accuracy' from parameter names
    parameter_names = results.columns.drop('accuracy')

    for stat in best_params.keys():
        # Group by each parameter and calculate the required statistic of 'accuracy'
        for param in parameter_names:
            grouped = results.groupby(param)['accuracy'].agg(['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'max']).rename(columns={'<lambda_0>': 'Q1', '<lambda_1>': 'Q3'})
            if stat in ['median', 'mean', 'max']:
                best_value = grouped[stat].idxmax()
            else:
                best_value = grouped[stat].idxmax()
            best_params[stat][param] = best_value

    return best_params


def evaluate_models(data: pd.DataFrame, best_params: Dict[str, Dict[str, Any]], random_state: int = 42) -> Dict[str, List[float]]:
    """
    Evaluates models based on provided best parameters and returns their performance.
    """
    results = {}
    # Iterate through each set of parameters (by statistic)
    for stat, params in best_params.items():
        model_accuracies = []
        for i in range(20):  # 20 different training/testing splits
            X_train, X_test, y_train, y_test = train_test_split(data.drop('cancer', axis=1), data['cancer'], test_size=0.20, random_state=random_state + i)
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            model_accuracies.append(accuracy)
        results[stat] = model_accuracies
    return results


def plot_evaluation_results_boxplot(evaluation_results: Dict[str, List[float]], output_file: Optional[str] = None) -> None:
    """
    Plots box plots for model evaluation results, showing the distribution of accuracies for different
    statistical parameter optimization strategies (e.g., median, mean, Q1, Q3, max).

    Parameters:
        evaluation_results (Dict[str, List[float]]): A dictionary where keys are the statistics used to
                                                     optimize parameters and values are lists of accuracies.
        output_file (Optional[str]): The path where the plot will be saved as a PDF file. If None, the plot
                                     will be shown on screen.

    Returns:
        None
    """
    # Create a DataFrame for easier plotting with seaborn
    data = {key: pd.Series(value) for key, value in evaluation_results.items()}
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title('Distribution of Model Accuracies by Parameter Optimization Strategy')
    plt.xlabel('Optimization Strategy')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Save or display the plot
    save_or_show_plot(output_file)


def select_best_optimization_strategy(evaluation_results: Dict[str, List[float]],
                                      best_params: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """
    Chooses the best optimization strategy based on the highest mean accuracy from evaluation results.

    Parameters:
        evaluation_results (Dict[str, List[float]]): A dictionary where keys are the statistics used to
                                                     optimize parameters (e.g., median, mean, Q1, Q3, max)
                                                     and values are lists of accuracies.
        best_params (Dict[str, Dict[str, Any]]): A dictionary containing the best parameters for each
                                                 optimization strategy.

    Returns:
        Tuple[Dict[str, Any], float]: A tuple containing the best parameters and the corresponding mean accuracy.
    """
    # Calculate the mean accuracy for each optimization strategy
    mean_accuracies = {stat: np.mean(acc) for stat, acc in evaluation_results.items()}

    # Determine the optimization strategy with the highest mean accuracy
    best_stat = max(mean_accuracies, key=mean_accuracies.get)
    best_accuracy = mean_accuracies[best_stat]
    best_parameters = best_params[best_stat]

    return (best_parameters, best_accuracy)


def train_final_model(data: pd.DataFrame, best_strategy: Dict[str, Any], save_path: str) -> None:
    """
    Trains a RandomForest model using the best strategy parameters and saves the model to the specified path.

    Parameters:
        data (pd.DataFrame): The complete dataset including features and the target.
        best_strategy (Dict[str, Any]): The best parameters for training the RandomForest model.
        save_path (str): The file path where the trained model will be saved.

    Returns:
        None
    """
    # Split the data into features and target
    X = data.drop('cancer', axis=1)
    y = data['cancer']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Initialize the RandomForest model with the best parameters
    model = RandomForestClassifier(**best_strategy)

    # Train the model
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Accuracy of the final model: {accuracy:.2f}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, save_path)
    logging.info(f"Model saved to {save_path}")

    # Save the best strategy parameters to a CSV file
    params_save_path = f"{save_path}_parameters.csv"
    pd.DataFrame([best_strategy]).to_csv(params_save_path, index=False)
    logging.info(f"Model parameters saved to {params_save_path}")


def main() -> None:
    """
    Main function to orchestrate the process of model training, saving results, and evaluation
    """
    args = parse_args()
    data = load_data(args.features_path)
    processed_data = preprocess_data(data)

    results = None  # Initialize results variable

    if os.path.exists(args.results_path):
        logging.info(f"Loading existing results from {args.results_path}")
        results = pd.read_csv(args.results_path)
    else:
        X_train, X_test, y_train, y_test = split_data(processed_data)
        results = train(X_train, y_train, X_test, y_test, save_path=args.models_path)
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        # Save the results DataFrame to a CSV file
        results.to_csv(args.results_path, index=False)
        logging.info(f"Results saved to {args.results_path}")

    # Graph results
    plot_heatmaps(results, f"{args.graphs_path}heatmap")
    plot_parameter_effects(results, f"{args.graphs_path}boxplots")
    plot_accuracy_vs_estimators(results, f"{args.graphs_path}accuracy_vs_n_estimators")

    # Select best parameters and evaluate the models for them
    best_parameters = select_best_parameters(results)
    evaluation_results = evaluate_models(processed_data, best_parameters)

    # Graph evaluation results
    plot_evaluation_results_boxplot(evaluation_results, f"{args.graphs_path}evaluation_results")

    # Select best strategy and train final model
    best_strategy, best_mean_accuracy = select_best_optimization_strategy(evaluation_results, best_parameters)
    logging.info(f"Best Strategy Parameters: {best_strategy}")
    logging.info(f"Best Mean Accuracy: {best_mean_accuracy}")
    train_final_model(processed_data, best_strategy, args.final_model_path)


if __name__ == "__main__":
    main()
