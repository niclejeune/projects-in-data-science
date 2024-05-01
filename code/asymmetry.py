import os
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_float
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.ndimage import center_of_mass, shift

def read_and_process_image(image_path):
    original_image = io.imread(image_path)
    if len(original_image.shape) == 3:
        gray_image = rgb2gray(original_image)
        thresh = threshold_otsu(gray_image)
        binary_image = gray_image > thresh
    else:
        binary_image = original_image > 0
    return binary_image

def calculate_asymmetry(binary_image):
    com = center_of_mass(binary_image)
    shift_needed = np.array(binary_image.shape) / 2.0 - np.array(com)
    shifted_image = shift(binary_image, shift_needed)
    shifted_image_float = img_as_float(shifted_image)
    return shifted_image_float

def measure_asymmetry(binary_img):
    vertical_flip = np.flip(binary_img, axis=0)
    horizontal_flip = np.flip(binary_img, axis=1)
    vertical_asymmetry = np.sum(np.logical_xor(binary_img, vertical_flip))
    horizontal_asymmetry = np.sum(np.logical_xor(binary_img, horizontal_flip))
    return vertical_asymmetry, horizontal_asymmetry

def process_images_in_directory(directory, output_path):
    asymmetry_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Ensuring to process only images
            f = os.path.join(directory, filename)
            binary_image = read_and_process_image(f)
            shifted_image_float = calculate_asymmetry(binary_image)
            results = {}
            for angle in range(0, 91, 5):
                rotated_image_float = transform.rotate(shifted_image_float, angle, resize=True, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
                rotated_image_binary = rotated_image_float > 0.5
                vertical, horizontal = measure_asymmetry(rotated_image_binary)
                results[angle] = (vertical, horizontal)
            for angle, (vertical, horizontal) in results.items():
                asymmetry_data.append([filename, angle, vertical, horizontal])
    save_results_to_csv(asymmetry_data, output_path)

def save_results_to_csv(data, file_path):
    df = pd.DataFrame(data, columns=['Filename', 'Angle', 'Vertical Asymmetry', 'Horizontal Asymmetry'])
    df.to_csv(file_path, index=False)

def main(input_directory, output_csv_file):
    process_images_in_directory(input_directory, output_csv_file)
