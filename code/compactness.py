import os
import numpy as np
from skimage import io
from skimage.measure import label, regionprops
import math

def calculate_compactness(mask_path):
    # Load the mask
    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[:, :, 0]  # Convert to grayscale if not already

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Label the image (in case there are multiple disjoint regions, consider the largest)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if not regions:
        return None  # No regions found

    # Assuming the largest region is the lesion of interest
    largest_region = max(regions, key=lambda x: x.area)
    area = largest_region.area
    perimeter = largest_region.perimeter

    # Calculate compactness
    compactness = 4 * math.pi * area / (perimeter ** 2)
    return compactness

def process_all_images(directory):
    compactness_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith("_mask.png"):
            full_path = os.path.join(directory, filename)
            compactness = calculate_compactness(full_path)
            if compactness is not None:
                compactness_dict[filename] = compactness
            else:
                compactness_dict[filename] = "No lesion found"
    return compactness_dict

# Usage
directory = 'data/masks/'  # Directory containing mask images
results = process_all_images(directory)
print(results)
