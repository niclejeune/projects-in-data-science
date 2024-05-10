import pandas as pd
import joblib
import argparse
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for specifying paths for feature data, final model, and results saving.

    Returns:
        argparse.Namespace: The namespace object containing all arguments.
    """
    parser = argparse.ArgumentParser(description="Predict cancer likelihood using a trained model and save the results.")
    parser.add_argument('--features_path', type=str, default='features/features.csv', help='Path to the features CSV file.')
    parser.add_argument('--model_path', type=str, default='training/final_model', help='Path to the trained model file.')
    parser.add_argument('--results_path', type=str, default='evaluation/results.csv', help='Path where the prediction results will be saved.')
    return parser.parse_args()

def load_and_prepare_data(features_path: str, include_labels: bool = False) -> pd.DataFrame:
    """
    Load and preprocess data from a specified file path.
    """
    data = pd.read_csv(features_path)
    features = ['hue', 'saturation', 'value', 'hsv_uniformity', 'compactness_score', 'Vertical Asymmetry_mean', 'Horizontal Asymmetry_mean']
    if include_labels:
        return data[['img_id'] + features + ['cancer']]
    else:
        return data[['img_id'] + features]

def predict_cancer_likelihood(model_path: str, data_path: str) -> pd.DataFrame:
    """
    Predict cancer likelihood scores and include cancer labels using a trained model.
    """
    model = joblib.load(model_path)
    data = load_and_prepare_data(data_path, include_labels=True)
    img_ids = data['img_id']
    features = data.drop(['img_id', 'cancer'], axis=1)
    predictions = model.predict_proba(features)[:, 1]
    results = pd.DataFrame({
        'img_id': img_ids,
        'cancer_likelihood': predictions,
        'cancer': data['cancer']
    })
    return results

def calculate_and_print_accuracy(results: pd.DataFrame) -> None:
    """
    Calculate and log the accuracy of predictions.
    """
    results['predicted_cancer'] = (results['cancer_likelihood'] > 0.50).astype(int)
    accuracy = (results['predicted_cancer'] == results['cancer']).mean()
    logging.info(f"Overall Accuracy (Threshold > 0.50): {accuracy:.2%}")

def main() -> None:
    """
    Main function to orchestrate the prediction process and save results.
    """
    args = parse_args()
    results = predict_cancer_likelihood(args.model_path, args.features_path)
    calculate_and_print_accuracy(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    # Save results
    results[['img_id', 'cancer_likelihood']].to_csv(args.results_path, index=False)
    logging.info(f"Results saved to {args.results_path}")

if __name__ == "__main__":
    main()
