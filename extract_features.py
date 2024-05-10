# Load libraries
from skimage import segmentation, color
from skimage.measure import label, regionprops
from skimage import io, transform, img_as_float
import matplotlib.pyplot as plt
import matplotlib.colors
import math
import pandas as pd
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.ndimage import center_of_mass, shift

#1st features: Color evenness
def calculate_hsv_deviations(hsv_image, mask):
    """
    Calculate the standard deviation for each HSV 
    channel within the masked area.
    """
    mask_bool = mask > 0 #mask_bool, has True values where the mask is greater than 0 (indicating the area of interest) and False elsewhere.
    std_devs = []
    deviations = []
    std_devs = {'Hue': 0, 'Saturation': 0, 'Value': 0, 'Uniformity_Score': 0}
    for i, channel_name in enumerate(['Hue', 'Saturation', 'Value']):  # Iterate over HUE, SATURATION, VALUE channels
        channel = hsv_image[:, :, i] # [x,y, H/S/V] where x and y coordinates
        masked_channel = channel[mask_bool] # Apply mask
        std_dev = np.std(masked_channel)
        deviations.append(std_dev)
        std_devs[channel_name] = std_dev
    std_devs['Uniformity_Score'] = np.mean(deviations)
    return std_devs

#for loop for processing all the pictures:
def process_images_with_hsv_uniformity(img_path, mask_path_o):
    '''
    This function is processing the images in HSV chanel to get 
    the standard deviation on the uniformity, gives a DF as result
    '''
    HSV_score = {'img_id':[],
                 'hue':[],
                 'saturation':[],
                 'value':[],
                'hsv_uniformity':[]}
    for filename in os.listdir(img_path):
        if filename.endswith(".png"):
            base_name = filename[:-4]
            image_path = os.path.join(img_path, filename)
            mask_name = base_name + '_mask.png'
            mask_path = os.path.join(mask_path_o, mask_name)
            
            rgb_img = plt.imread(image_path)[:,:,:3]
            mask = plt.imread(mask_path)
            
            if filename is None or mask_name is None:
                HSV_score['img_id'].append(filename)
                HSV_score['hue','saturation','value','hsv_uniformity'].append('N/A')
                continue
                
            #Average colour on the lesion:
            img_avg_lesion = rgb_img.copy()
            for i in range(3):
                channel = img_avg_lesion[:,:,i]
                mean = np.mean(channel[mask.astype(bool)])
                channel[mask.astype(bool)] = mean
                img_avg_lesion[:,:,i] = channel

            # cropping the image according to the mask
            lesion_coords = np.where(mask != 0)
            min_x = min(lesion_coords[0])
            max_x = max(lesion_coords[0])
            min_y = min(lesion_coords[1])
            max_y = max(lesion_coords[1])
            cropped_lesion = rgb_img[min_x:max_x,min_y:max_y]

            # Segment the cropped lesion with SLIC (without boundaries)
            img_avg_lesion_1 = rgb_img.copy()
            segments_1 = segmentation.slic(cropped_lesion, compactness = 50, n_segments=10, sigma=3,
                                        start_label=1)
            out_1 = color.label2rgb(segments_1, cropped_lesion, kind='avg')
            img_avg_lesion_1[min_x:max_x,min_y:max_y] = out_1
            img_avg_lesion_1[mask == 0] = rgb_img[mask==0]            

            img_avg_lesion_1_hsv = matplotlib.colors.rgb_to_hsv(img_avg_lesion_1)
            
            uniformity = calculate_hsv_deviations(img_avg_lesion_1_hsv, mask)
            HSV_score['img_id'].append(filename)
            HSV_score['hue'].append(uniformity['Hue'])
            HSV_score['saturation'].append(uniformity['Saturation'])
            HSV_score['value'].append(uniformity['Value'])
            HSV_score['hsv_uniformity'].append(uniformity['Uniformity_Score'])

    HSV_score_df = pd.DataFrame(HSV_score)
    return  HSV_score_df


# 2nd feature: Compactness
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
    compactness_dict = {'img_id':[],
                 'compactness_score':[]}
    for filename in os.listdir(directory):
        if filename.endswith("_mask.png"):
            full_path = os.path.join(directory, filename)
            compactness = calculate_compactness(full_path)
            # I need to remove the _mask from the file names
            filename = filename.replace("_mask", "")
            if compactness is not None:
                compactness_dict['img_id'].append(filename)
                compactness_dict['compactness_score'].append(compactness)
            else:
                compactness_dict['img_id'].append(filename)
                compactness_dict['compactness_score'].append('N/A')
    compactness_score_df = pd.DataFrame(compactness_dict)
    return compactness_score_df


#3rd features: Assymetry
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

def process_images_in_directory(directory):
    """
       Processes all PNG images in the specified directory to measure their asymmetry.
       Args:
           directory (str): Path to the directory containing images.
           output_path (str): Path where the CSV output will be saved.
       """
    asymmetry_data = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Ensuring to process only images
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
            for angle, (vertical, horizontal) in results.items():
                # Store results for each angle
                asymmetry_data.append([filename, angle, vertical, horizontal])
    return data_framing(asymmetry_data)

def data_framing(data):
    df = pd.DataFrame(data, columns=['Filename', 'Angle', 'Vertical Asymmetry', 'Horizontal Asymmetry'])
    assymetry_df = preprocess_assymetry(df)
    return assymetry_df

def main_assy_fun(input_directory):
    assymetry_df = process_images_in_directory(input_directory)
    return assymetry_df


# Pre-process data from assymetry: 
def preprocess_assymetry(data):

    ''' This function standardize the Assymetry measurments, and 
    creats a datafram out of it - for creating a features csv file.
    THIS IS A HELPER FUNCTION'''

    # Remove '_mask' from filenames
    data['Filename'] = data['Filename'].str.replace('_mask.png', '.png')
    data = data.rename(columns={'Filename': 'img_id'})

    # Drop the 'Mask' column
    data = data.drop(columns=['Angle'])
    data.columns = ["img_id", "Vertical Asymmetry_mean", "Horizontal Asymmetry_mean"]

    # Calculate mean for each image
    summary_df =  data.groupby("img_id").mean()
    return summary_df


