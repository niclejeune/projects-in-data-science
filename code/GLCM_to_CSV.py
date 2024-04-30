##This is a duplicated py file, USE THIS!

import evenness
import GLCM
import pandas as pd

# Using realitve path to navigate to the images and the masks and creating a DF from the meta data csv for the analisys
img = './data/images'
mask = './data/masks'
meta_csv = './additional_files/metadata.csv'
meta_data = pd.read_csv(meta_csv)
meta_data = meta_data[['img_id','diagnostic','smoke']]

# Processing the images with evenness and GLCM (and the other features too)
evenness_score = evenness.process_images_with_hsv_uniformity(img,mask)
GLCM_score = GLCM.process_images_with_glcm(img,mask)

#Creating DFs from our data and combine it with the diagnostics, and smokers
evenness_df = pd.DataFrame(evenness_score)
GLCM_df = pd.DataFrame(GLCM_score)
combined_df = pd.merge(meta_data,evenness_df, on='img_id')
combined_df = pd.merge(combined_df,GLCM_df, on='img_id')
combined_df.to_csv('img_id_and_values_test_with123.csv', index=False)
