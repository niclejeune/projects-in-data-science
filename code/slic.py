import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
from skimage import io, segmentation
from sklearn.cluster import KMeans

def process(image_path, lesion_masks_dir, output_mask_dir, output_overlay_dir, n_clusters=5):
    """
    Process an image with its lesion mask, colorize clusters within the lesion,
    save masks of these clusters, and create an overlay of the colorized clusters
    on the original image.
    """
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
    colors = [tuple(np.random.choice(range(256), size=3)) for _ in range(n_clusters)]  # Generate unique colors

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
        cluster_mask_filename = f"{os.path.splitext(base_filename)[0]}_lesion_cluster_{i}{os.path.splitext(base_filename)[1]}"
        cv2.imwrite(os.path.join(output_mask_dir, cluster_mask_filename), mask)
        
        # Create and save the overlay
        overlay_image = cv2.addWeighted(overlay_image, 1, color_mask, 0.5, 0)
    
    overlay_filename = f"{os.path.splitext(base_filename)[0]}_overlay{os.path.splitext(base_filename)[1]}"
    cv2.imwrite(os.path.join(output_overlay_dir, overlay_filename), overlay_image)

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

#Directories path :
dataset_dir = "..."
lesion_masks_dir = "..."
output_mask_dir = "..."
output_overlay_dir = "..."

if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)
if not os.path.exists(output_overlay_dir):
    os.makedirs(output_overlay_dir)

for filename in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, filename)
    process(image_path, lesion_masks_dir, output_mask_dir, output_overlay_dir)

print("Processing complete. Masks and overlays saved.")
