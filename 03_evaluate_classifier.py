import joblib
import numpy as np
from extract_features import extract_features  # Assuming this function is defined elsewhere


def classify(img, mask):
    """
    Classify an image as cancerous or non-cancerous based on features extracted from the image and a mask.

    Args:
        img (numpy.ndarray): The image da`ta as a numpy array.
        mask (numpy.ndarray): The mask data as a numpy array.

    Returns:
        tuple: A tuple containing the predicted label ('cancerous' or 'non-cancerous') and the probability of cancer.
    """
    # Extract features from the image and mask
    features = extract_features(img, mask)

    # Load the pre-trained classifier model
    model_path = 'training/final_model'  # Update the path as needed
    classifier = joblib.load(model_path)

    # Predict the probability of cancer
    pred_prob = classifier.predict_proba([features])[:, 1][0]

    # Determine the label based on the probability
    label = 'cancerous' if pred_prob > 0.5 else 'non-cancerous'

    return label, pred_prob
