import os
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_float
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.ndimage import center_of_mass, shift

def read_and_process_image(image_path):
    """
    Reads an image from the specified path and processes it into a binary image based on a threshold.
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Binary image where pixels are either True (1) or False (0).
    """
    # Read the image from the file
    original_image = io.imread(image_path)
    
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

def shift_to_center(binary_image):
    """
    Calculates the center of mass of the binary image and shifts it to the center of the image.
    Args:
        binary_image (numpy.ndarray): Binary image.
    
    Returns:
        numpy.ndarray: Shifted binary image as float values.
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

def measure_asymmetry(binary_img):
    """
    Measures the asymmetry of the image by flipping it vertically and horizontally.
    Args:
        binary_img (numpy.ndarray): Binary image.
    
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

def process_images_in_directory(directory, output_path):
    """
    Processes all PNG images in the specified directory to measure their asymmetry.
    Args:
        directory (str): Path to the directory containing images.
        output_path (str): Path where the CSV output will be saved.
    """
    asymmetry_data = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Process only PNG images
        if filename.endswith(".png"):
            # Construct the full file path
            f = os.path.join(directory, filename)
            # Read and process the image to binary
            binary_image = read_and_process_image(f)
            # Calculate center of mass and shift the image
            shifted_image_float = shift_to_center(binary_image)
            results = {}
            # Rotate the image at 5-degree intervals up to 90 degrees
            for angle in range(0, 91, 5):
                rotated_image_float = transform.rotate(shifted_image_float, angle, resize=True, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
                rotated_image_binary = rotated_image_float > 0.5
                # Measure asymmetry for each rotated image
                vertical, horizontal = measure_asymmetry(rotated_image_binary)
                results[angle] = (vertical, horizontal)
            # Store results for each angle
            for angle, (vertical, horizontal) in results.items():
                asymmetry_data.append([filename, angle, vertical, horizontal])
    # Save the results to a CSV file
    save_results_to_csv(asymmetry_data, output_path)

def save_results_to_csv(data, file_path):
    """
    Saves the collected asymmetry data into a CSV file.
    Args:
        data (list): List of lists containing filename, angle, and asymmetry data.
        file_path (str): Path to the CSV file where data will be saved.
    """
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['Filename', 'Angle', 'Vertical Asymmetry', 'Horizontal Asymmetry'])
    # Save the DataFrame to a CSV file without the index
    df.to_csv(file_path, index=False)

def main(input_directory, output_csv_file):
    """
    Main function to process all images in a directory and save the results.
    Args:
        input_directory (str): Directory containing images to process.
        output_csv_file (str): Path to the CSV file where results will be saved.
    """
    # Process all images in the specified directory
    process_images_in_directory(input_directory, output_csv_file)
