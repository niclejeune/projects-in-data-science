import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    return parser.parse_args()


def classify(img: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Classify an image as cancerous or non-cancerous based on features extracted from the image and a mask.

    This function performs the classification by extracting features from the provided image and mask,
    utilizing a pre-trained model to predict the likelihood of cancer presence. It returns both a label
    ('cancerous' or 'non-cancerous') and the probability of the cancer prediction.

    Args:
        img (np.ndarray): The image data as a numpy array, typically with dimensions corresponding to the height, width, and color channels.
        mask (np.ndarray): The mask data as a numpy array, typically with dimensions corresponding to the height and width, used to identify relevant areas in the image.

    Returns:
        tuple: A tuple containing the predicted label ('cancerous' or 'non-cancerous') and the probability of cancer.
    """
    # Extract features from the image and mask
    feature_values = extract_features(img, mask)

    # Define feature names explicitly
    feature_names = ['hue', 'saturation', 'value', 'hsv_uniformity', 'compactness_score', 'Vertical Asymmetry_mean',
                     'Horizontal Asymmetry_mean']

    # Create a DataFrame for the features, with the defined feature names
    features_df = pd.DataFrame([feature_values], columns=feature_names)

    # Load the classifier model
    model_path = 'training/final_model'
    classifier = joblib.load(model_path)

    # Predict the probability of cancer
    pred_prob = classifier.predict_proba(features_df)[:, 1][0]

    # Determine the label based on the probability
    label = 'cancerous' if pred_prob > 0.5 else 'non-cancerous'

    return label, pred_prob


def main() -> None:
    """
        Main function to iterate over entire directory and classify all images within
    """
    args = parse_args()
    for image_filename in os.listdir(args.images_path):
        if image_filename.endswith(".png"):
            base_name = image_filename[:-4]
            image_path = os.path.join(args.images_path, image_filename)
            mask_path = os.path.join(args.masks_path, base_name + '_mask.png')

            image = plt.imread(image_path)[:, :, :3]
            mask = plt.imread(mask_path)
            label, pred_prob = classify(image, mask)
            logging.info(f"Image {base_name} - {pred_prob} - {pred_prob * 100:.2f}%")


if __name__ == '__main__':
    main()
