from typing import List

import numpy as np
from scipy.ndimage import center_of_mass, shift
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage import color, transform, segmentation
import matplotlib.colors as colors
from skimage.filters import threshold_otsu
from skimage.util import img_as_float
import math


def extract_features(image: np.ndarray, mask: np.ndarray) -> List[float]:
    """
    Extracts individual HSV channel values, uniformity, compactness, and asymmetry features from a given image and its corresponding mask.
    This method leverages multiple image processing techniques to evaluate various characteristics of the lesion area identified by the mask.

    Args:
        image (np.ndarray): Image data as a numpy array.
        mask (np.ndarray): Mask data as a binary numpy array.

    Returns:
        List[float]: List containing extracted features [hue, saturation, value, hsv_uniformity, compactness, vertical_asymmetry, horizontal_asymmetry].
    """
    # HSV
    hue, saturation, value, hsv_uniformity = process_images_with_hsv_uniformity(image, mask)

    # Compactness calculation
    compactness = calculate_compactness(mask)

    # Asymmetry calculation
    vertical_asymmetry, horizontal_asymmetry = process_mask_asymmetry(mask)

    return [hue, saturation, value, hsv_uniformity, compactness, vertical_asymmetry, horizontal_asymmetry]


# HSV and uniformity


def calculate_hsv_deviations(hsv_image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calculate the standard deviation for each HSV channel within the masked area.
    This function applies a binary mask to an HSV image and calculates the standard deviation for each channel,
    which is used to derive a uniformity score for the lesion area.

    Args:
        hsv_image (np.ndarray): HSV image data.
        mask (np.ndarray): Mask data as a binary array.

    Returns:
        dict: Dictionary containing the standard deviations of the 'Hue', 'Saturation', 'Value' channels and their mean (uniformity score).
    """
    mask_bool = mask > 0
    deviations = []
    std_devs = {'Hue': 0, 'Saturation': 0, 'Value': 0}

    for i, channel_name in enumerate(['Hue', 'Saturation', 'Value']):
        channel = hsv_image[:, :, i]
        masked_channel = channel[mask_bool]
        std_dev = np.std(masked_channel)
        std_devs[channel_name] = std_dev
        deviations.append(std_dev)

    std_devs['Uniformity'] = np.mean(deviations)  # Calculate the mean of deviations as the uniformity score
    return std_devs


def process_images_with_hsv_uniformity(image, mask):
    """
    Processes a single image and its mask by averaging colors in lesion, segmenting and calculating HSV deviations including uniformity.
    Args:
        image (numpy.ndarray): Image data as a numpy array.
        mask (numpy.ndarray): Mask data as a numpy array.

    Returns:
        tuple: Tuple containing the standard deviations of 'hue', 'saturation', 'value', and their mean uniformity.
    """
    # Ensure only the RGB channels are considered if extra channels exist
    rgb_img = image[:, :, :3]

    # Average colour on the lesion:
    img_avg_lesion = rgb_img.copy()
    for i in range(3):
        channel = img_avg_lesion[:, :, i]
        mean = np.mean(channel[mask.astype(bool)])
        channel[mask.astype(bool)] = mean
        img_avg_lesion[:, :, i] = channel

    # Cropping the image according to the mask
    lesion_coords = np.where(mask != 0)
    min_x = min(lesion_coords[0])
    max_x = max(lesion_coords[0])
    min_y = min(lesion_coords[1])
    max_y = max(lesion_coords[1])
    cropped_lesion = rgb_img[min_x:max_x, min_y:max_y]

    # Segment the cropped lesion with SLIC (without boundaries)
    segments = segmentation.slic(cropped_lesion, compactness=50, n_segments=10, sigma=3, start_label=1)
    out = color.label2rgb(segments, cropped_lesion, kind='avg')
    img_avg_lesion[min_x:max_x, min_y:max_y] = out
    img_avg_lesion[mask == 0] = rgb_img[mask == 0]

    # Convert the modified image to HSV
    img_avg_lesion_hsv = colors.rgb_to_hsv(img_avg_lesion)

    # Calculate HSV deviations
    std_devs = calculate_hsv_deviations(img_avg_lesion_hsv, mask)

    # Extracting values from the dictionary to return as a tuple
    return (std_devs['Hue'], std_devs['Saturation'], std_devs['Value'], std_devs['Uniformity'])


# Compactness


def calculate_compactness(mask: np.ndarray) -> float:
    """
    Calculate the compactness of the largest lesion region identified in the binary mask.
    Compactness is defined as the ratio of the area to the square of the perimeter.

    Args:
        mask (np.ndarray): Mask data as a binary array.

    Returns:
        float: Compactness score of the largest region.
    """
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


# Asymmetry


def process_mask_asymmetry(mask: np.ndarray) -> tuple:
    """
    Measure the asymmetry of the lesion in the mask. The function computes asymmetry scores by rotating the lesion
    and comparing differences between the original and flipped images at various angles.

    Args:
        mask (np.ndarray): Mask data as a binary array.

    Returns:
        tuple: Mean vertical and horizontal asymmetry scores.
    """
    binary_image = process_image(mask)
    # Calculate center of mass and shift the image
    shifted_image_float = shift_to_center(binary_image)

    vertical_asymmetries = []
    horizontal_asymmetries = []
    # Rotate the image at 5-degree intervals up to 90 degrees
    for angle in range(0, 91, 5):
        rotated_image_float = transform.rotate(shifted_image_float, angle, resize=True, center=None, order=1,
                                               mode='constant', cval=0, clip=True, preserve_range=True)
        rotated_image_binary = rotated_image_float > 0.5
        # Measure asymmetry for each rotated image
        vertical, horizontal = measure_asymmetry(rotated_image_binary)
        vertical_asymmetries.append(vertical)
        horizontal_asymmetries.append(horizontal)

    mean_vertical_asymmetry = np.mean(vertical_asymmetries)
    mean_horizontal_asymmetry = np.mean(horizontal_asymmetries)

    return mean_vertical_asymmetry, mean_horizontal_asymmetry


def process_image(original_image: np.ndarray) -> np.ndarray:
    """
    Converts an image to a binary format based on optimal thresholding.
    This processing is for further analysis of asymmetry.

    Args:
        original_image (np.ndarray): Original image data.

    Returns:
        np.ndarray: Binary image where pixels are either True (1) or False (0).
    """
    # Check if the image has three dimensions (color image)
    if len(original_image.shape) == 3:
        # Convert the image to grayscale
        gray_image = rgb2gray(original_image)
        # Apply Otsu's method to find an optimal threshold value for binarization
        thresh = threshold_otsu(gray_image)
        # Create a binary image by applying the threshold
        binary_image = gray_image > thresh
    else:
        # For already grayscale images, simply threshold at zero
        binary_image = original_image > 0

    return binary_image


def shift_to_center(binary_image: np.ndarray) -> np.ndarray:
    """
    Shift the center of mass of a binary image to its geometric center. This adjustment is needed for proper asymmetry analysis.

    Args:
        binary_image (np.ndarray): Binary image.

    Returns:
        np.ndarray: Shifted binary image as float values.
    """
    # Calculate the center of mass of the binary image
    com = center_of_mass(binary_image)
    # Calculate the shift needed to center the mass at the image center
    shift_needed = np.array(binary_image.shape) / 2.0 - np.array(com)
    # Shift the image so the center of mass is at the geometric center
    shifted_image = shift(binary_image, shift_needed)
    # Convert the shifted binary image to floating point format
    shifted_image_float = img_as_float(shifted_image)

    return shifted_image_float


def measure_asymmetry(binary_img: np.ndarray) -> tuple:
    """
    Measures the asymmetry of the image by flipping it vertically and horizontally and calculating the discrepancy from the original.

    Args:
        binary_img (np.ndarray): Binary image.

    Returns:
        tuple: Vertical and horizontal asymmetry scores.
    """
    # Flip the image vertically
    vertical_flip = np.flip(binary_img, axis=0)
    # Flip the image horizontally
    horizontal_flip = np.flip(binary_img, axis=1)
    # Calculate vertical asymmetry by comparing the original and vertically flipped images
    vertical_asymmetry = np.sum(np.logical_xor(binary_img, vertical_flip))
    # Calculate horizontal asymmetry by comparing the original and horizontally flipped images
    horizontal_asymmetry = np.sum(np.logical_xor(binary_img, horizontal_flip))

    return vertical_asymmetry, horizontal_asymmetry
