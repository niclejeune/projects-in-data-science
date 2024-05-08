from skimage import segmentation, color
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib.colors
import math
import numpy as np
import os
import pandas as pd

def calculate_hsv_deviations(hsv_image, mask):
    """Calculate the standard deviation for each HSV channel within the masked area."""
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
#    std_devs['Uniformity_Score'] = np.mean(deviations) useless
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
            
            # original_image = cv2.imread(image_path) #doesn't work with SLIC implementation
            # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #doesn't work with SLIC implementation
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
# SLIC
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

            img_avg_lesion_1_hsv = matplotlib.colors.rgb_to_hsv(img_avg_lesion_1)             #rgb to hsv
            
            uniformity = calculate_hsv_deviations(img_avg_lesion_1_hsv, mask)
            HSV_score['img_id'].append(filename)
            HSV_score['hue'].append(uniformity['Hue'])
            HSV_score['saturation'].append(uniformity['Saturation'])
            HSV_score['value'].append(uniformity['Value'])
           # HSV_score['hsv_uniformity'].append(uniformity['Uniformity_Score']) removing because it is useless

    HSV_score_df = pd.DataFrame(HSV_score)
    return  HSV_score_df

