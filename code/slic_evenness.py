import os
os.environ["OMP_NUM_THREADS"] = "1" #for windows users, remove the memory leak warning
import csv
import cv2
import numpy as np
from skimage import io, segmentation
from sklearn.cluster import KMeans

def extract_features(image, lesion_mask):
    """
    Extracts features from an image using SLIC segmentation.
    """
    segments_slic = segmentation.slic(image, n_segments=100, compactness=10, sigma=1, start_label=1, mask=lesion_mask)
    mean_colors = []
    for segVal in np.unique(segments_slic):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments_slic == segVal] = True
        mean_color = cv2.mean(image, mask=mask)[:3]
        mean_colors.append(mean_color)
    return segments_slic, np.array(mean_colors)

def calculate_hsv(hsv_image, mask):
    """
    Calculate the standard deviation and uniformity score based on HSV within the masked area.
    """
    mask_bool = mask > 0
    std_devs = []
    for i in range(3):  # Iterate over H, S, V channels
        channel = hsv_image[:, :, i]
        masked_channel = channel[mask_bool]  # Apply mask
        if len(masked_channel) > 1:
            std_dev = np.std(masked_channel)
        else:
            std_dev = 0  # If only one value in the masked region, set standard deviation to 0
        std_devs.append(std_dev)
    uniformity_score = np.nanmean(std_devs)  # Use np.nanmean to handle NaN values
    return uniformity_score, std_devs

def process(image_path, lesion_masks_dir, n_clusters=5, output_mask_dir="slic_masks", output_overlay_dir="slic_overlays"):
    """
    Process an image with its lesion mask, save masks of slic clusters and create an overlay of the colorized clusters on the original image.
    """
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    if not os.path.exists(output_overlay_dir):
        os.makedirs(output_overlay_dir)
    image = io.imread(image_path)
    base_filename = os.path.basename(image_path)
    mask_filename = f"{os.path.splitext(base_filename)[0]}_mask{os.path.splitext(base_filename)[1]}"
    lesion_mask_path = os.path.join(lesion_masks_dir, mask_filename)
    lesion_mask = io.imread(lesion_mask_path) > 0
    
    image_masked = image.copy()
    for i in range(3):
        image_masked[:, :, i] = image_masked[:, :, i] * lesion_mask

    segments_slic, mean_colors = extract_features(image_masked, lesion_mask)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(mean_colors)
    labels = kmeans.labels_
    
    overlay_image = image.copy()

    primary_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    colors = primary_colors[:n_clusters]

    for i in range(n_clusters):
        mask = np.zeros(segments_slic.shape, dtype=np.uint8)
        mask[np.isin(segments_slic, np.where(labels == i)[0])] = 255
        mask[~lesion_mask] = 0

        # Check the number of channels in the original image to handle both RGB and RGBA images
        num_channels = image.shape[2] if len(image.shape) > 2 else 1  # Default to 1 if the image is grayscale
        color_mask = np.zeros((image.shape[0], image.shape[1], num_channels), dtype=np.uint8)  # Match the original image's channels

        # Prepare the color accordingly
        if num_channels == 4:  # RGBA
            color = colors[i] + (255,)  # Add full opacity for the alpha channel
        else:  # RGB or grayscale
            color = colors[i]

        color_mask[mask == 255] = color

        # Ensure overlay_image has the same number of channels as color_mask
        if overlay_image.shape[2] != color_mask.shape[2]:
            if overlay_image.shape[2] == 3 and color_mask.shape[2] == 4:
                overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2RGBA)
            elif overlay_image.shape[2] == 4 and color_mask.shape[2] == 3:
                color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2RGBA)

        # Save the mask
        cluster_mask_filename = f"{os.path.splitext(base_filename)[0]}_segment_{i}{os.path.splitext(base_filename)[1]}"
        cv2.imwrite(os.path.join(output_mask_dir, cluster_mask_filename), mask)
        
        # Create and save the overlay
        overlay_image = cv2.addWeighted(overlay_image, 1, color_mask, 0.5, 0)
    
    overlay_filename = f"{os.path.splitext(base_filename)[0]}_overlay{os.path.splitext(base_filename)[1]}"
    cv2.imwrite(os.path.join(output_overlay_dir, overlay_filename), overlay_image)

def get_number_of_masks(image_file, mask_dir):
    """
    Get the number of masks associated with an image.
    """
    image_name = os.path.splitext(image_file)[0]
    mask_files = [file for file in os.listdir(mask_dir) if file.startswith(image_name + "_segment_")]
    return len(mask_files)

def apply_evenness_to_dataset(image_dir, mask_dir="slic_masks", output_file="hsv.csv"):
    """
    Apply HSV uniformity calculation to all images and their masks (slic clusters) in the dataset and write results to a CSV file.
    """
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['Image', 'Mask', 'Uniformity Score', 'Std Dev Hue', 'Std Dev Saturation', 'Std Dev Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        image_files = os.listdir(image_dir)
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            num_masks = get_number_of_masks(image_file, mask_dir)
            
            for i in range(num_masks):
                mask_file = f"{os.path.splitext(image_file)[0]}_segment_{i}.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    uniformity_score, (std_dev_hue, std_dev_saturation, std_dev_value) = calculate_hsv(hsv_image, mask)
                    writer.writerow({'Image': image_file, 'Mask': i, 'Uniformity Score': uniformity_score, 
                                     'Std Dev Hue': std_dev_hue, 'Std Dev Saturation': std_dev_saturation, 'Std Dev Value': std_dev_value})

def main(dataset_dir, dataset_mask):
    for filename in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, filename)
        process(image_path, dataset_mask, 4)

    apply_evenness_to_dataset(dataset_dir)
    print("Processing complete. Masks and overlays saved.")
