#it's a good idea to make a plot to see how they are distributed
import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(gray_image, mask):
    """Calculate GLCM features for the masked area of a grayscale image."""
    mask_bool = mask > 0
    if np.sum(mask_bool) == 0:
        return 0, 0, 0, 0

    masked_image = gray_image[mask_bool]
    if masked_image.ndim != 1:
        raise ValueError("Masked image dimension error. Expecting a 1D array of pixel values.")
    masked_image = np.clip(masked_image, 0, 255).astype('uint8')
    glcm = graycomatrix(masked_image.reshape(-1, 1), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    
    return np.mean(contrast), np.mean(homogeneity), np.mean(energy), np.mean(correlation)

def find_mask(mask_dirs, base_name):
    """Recursively find the first available mask for a given image base name from directory and subdirectories."""
    for root, dirs, files in os.walk(mask_dirs):
        for file in files:
            if file == base_name + '_mask.png':
                return cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
    return None

def process_images_with_glcm(data_dirs, mask_dir):
    GLMC_score = {'img_id':[],
                 'contrast':[],
                 'homogeneity':[],
                 'energy':[],
                 'correlation':[]}
    
    for filename in os.listdir(data_dirs):
        if filename.endswith(".png"):
            base_name = filename[:-4]
            image_path = os.path.join(data_dirs, filename)
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error loading image {filename}. Skipping...")
                continue

            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            mask = find_mask(mask_dir, base_name)
            if mask is None:
                GLMC_score['img_id'].append(filename)
                GLMC_score['contrast'].append('N/A')
                GLMC_score['homogeneity'].append('N/A')
                GLMC_score['energy'].append('N/A')
                GLMC_score['correlation'].append('N/A')
                continue

            contrast, homogeneity, energy, correlation = calculate_glcm_features(gray_image, mask)
            GLMC_score['img_id'].append(filename)
            GLMC_score['contrast'].append(contrast)
            GLMC_score['homogeneity'].append(homogeneity)
            GLMC_score['energy'].append(energy)
            GLMC_score['correlation'].append(correlation)

    return GLMC_score
             

# 'Brief description of sub-features
# 1. Contrast
# What It Measures: Difference in intensity between a pixel and its neighbors across the image.
# Interpretation: Higher values indicate a textured or highly variable image area.
# 2. Homogeneity
# What It Measures: How close the elements of the GLCM are to its diagonal.
# Interpretation: Higher values suggest smooth and less varied texture.
# 3. Energy
# What It Measures: Sum of squared values in the GLCM.
# Interpretation: Higher values denote uniform texture and regular patterns.
# 4. Correlation
# What It Measures: Degree to which a pixel is correlated to its neighbors.
# Interpretation: High values imply predictable patterns and structured textures.'
