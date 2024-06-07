import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import common
from extract_features import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(description="Classify images.")
    parser.add_argument('--images_path', type=str, default='data/images', help='Path to the images folder')
    parser.add_argument('--masks_path', type=str, default='data/masks', help='Path to the masks folder')
    parser.add_argument('--features_path', type=str, default='', help='Path to the features csv, for comparing predicted to actual')
    parser.add_argument('--output_path', type=str, default='graphs/confusion.png', help='Path to save confusion matrix')
    parser.add_argument('--roc_path', type=str, default='graphs/roc.png', help='Path to save ROC curve')
    return parser.parse_args()


def classify(img: np.ndarray, mask: np.ndarray, model=None) -> tuple:
    """
    Classify an image as cancerous or non-cancerous based on features extracted from the image and a mask.

    This function performs the classification by extracting features from the provided image and mask,
    utilizing a pre-trained model to predict the likelihood of cancer presence. It returns both a label
    ('cancerous' or 'non-cancerous') and the probability of the cancer prediction.

    Args:
        img (np.ndarray): The image data as a numpy array, typically with dimensions corresponding to the height, width, and color channels.
        mask (np.ndarray): The mask data as a numpy array, typically with dimensions corresponding to the height and width, used to identify relevant areas in the image.
        model: The pre-trained classifier model. If None, the default final model will be loaded within the function.

    Returns:
        tuple: A tuple containing the predicted label ('cancerous' or 'non-cancerous') and the probability of cancer.
    """
    # Extract features from the image and mask
    feature_values = extract_features(img, mask)

    # Define feature names explicitly
    feature_names = ['hue', 'saturation', 'value', 'compactness_score', 'Vertical Asymmetry_mean',
                     'Horizontal Asymmetry_mean', 'dots']

    # Create a DataFrame for the features, with the defined feature names
    features_df = pd.DataFrame([feature_values], columns=feature_names)

    # Load the classifier model if not provided
    if model is None:
        model_path = 'training/final_model'
        model = joblib.load(model_path)

    # Predict the probability of cancer
    pred_prob = model.predict_proba(features_df)[:, 1][0]

    # Determine the label based on the probability
    label = 'cancerous' if pred_prob > 0.5 else 'non-cancerous'

    return label, pred_prob


def main() -> None:
    """
    Main function to iterate over entire directory and classify all images within.
    """
    args = parse_args()

    # Load the classifier model once and pass it to classify function
    classifier_model = joblib.load('training/final_model')

    # Load the features CSV if provided
    if args.features_path:
        features_df = pd.read_csv(args.features_path)
        features_df.set_index('img_id', inplace=True)
    else:
        features_df = None

    y_true = []
    y_pred = []
    y_prob = []

    for image_filename in os.listdir(args.images_path):
        if image_filename.endswith(".png"):
            base_name = image_filename[:-4]
            image_path = os.path.join(args.images_path, image_filename)
            mask_path = os.path.join(args.masks_path, base_name + '_mask.png')

            image = plt.imread(image_path)[:, :, :3]
            mask = plt.imread(mask_path)
            label, pred_prob = classify(image, mask, classifier_model)

            if features_df is not None and image_filename in features_df.index:
                actual_cancer = features_df.loc[image_filename, 'cancer']
                actual_label = 'cancerous' if actual_cancer == 1 else 'non-cancerous'
                match = 'MATCH' if label == actual_label else 'MISMATCH'
                logging.info(f"Image {base_name} - Predicted: {label}, Actual: {actual_label} - {match}")

                y_true.append(actual_label)
                y_pred.append(label)
                y_prob.append(pred_prob)
            else:
                logging.info(f"Image {base_name} - {label} - {pred_prob * 100:.2f}%")

    if y_true and y_pred:
        # Generate and display confusion matrix
        labels = ['non-cancerous', 'cancerous']
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        common.save_or_show_plot(args.output_path)

        # Generate and save ROC curve
        y_true_binary = [1 if label == 'cancerous' else 0 for label in y_true]
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        common.save_or_show_plot(args.roc_path)


if __name__ == '__main__':
    main()
