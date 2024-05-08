# Load libraries
from skimage import segmentation, color
from skimage.measure import label, regionprops
from skimage import io, transform, img_as_float
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
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
import extract_features

def creating_basic_df(meta_data):
    '''
    This function is preparing the main df for the features csv
    '''
    meta_data = pd.read_csv(meta_csv)
    meta_data = meta_data[['img_id','diagnostic']]
    meta_data['cancer'] = 0
    meta_data['cancer'] = meta_data['cancer'].fillna(0)
    meta_data.loc[meta_data['diagnostic'].isin(['BCC', 'SCC', 'MEL']), 'cancer'] = 1
    return meta_data

# Processing the images with evenness and GLCM (and the other features too)
def creating_missing_df_img(img,mask):
    '''
    This function is creating DFs from the data from various features 
    for creating the features csv
    '''
    print('Processing evenness data...')
    evenness_score = extract_features.process_images_with_hsv_uniformity(img,mask)
    evenness_df = pd.DataFrame(evenness_score)
    print('Processing compactness data...')
    compactness_score = extract_features.process_all_images(mask)
    compactness_df = pd.DataFrame(compactness_score)
    print('Processing assymetry data...')
    asssymetry_score = extract_features.main_assy_fun(mask)
    asssymetry_df = pd.DataFrame(asssymetry_score)
    return evenness_df, compactness_df, asssymetry_df

def merging_data_frames(meta_df,even_df,compact_df,assy_df):
    '''
    This function merges the data frames together, and saves it as csv
    '''
    print('Combining data frames and saving to CSV')
    combined_df = pd.merge(meta_df,even_df, on='img_id')
    combined_df_2 = pd.merge(combined_df,compact_df, on = 'img_id')
    combined_df_3 = pd.merge(combined_df_2,assy_df, on = 'img_id')
    combined_df_3.to_csv('./exam_file_structure/features/features.csv', index=False)
    message = 'CSV file saved!'
    print(message)

# Using realitve path to navigate to the images and the masks and creating a DF from the meta data csv for the analisys
img = 'Masks1300images/Images'
mask = 'Masks1300images/Masks'
meta_csv = 'Masks1300images/1300-IMAGES-CSV-FILE.csv'

meta_data_df = creating_basic_df(meta_csv)
evenness_df, compactness_df, assymetry_df = creating_missing_df_img(img,mask)

merging_data_frames(meta_data_df,evenness_df,compactness_df,assymetry_df)

# print(f'This is evennes:{evenness_df}')
# print(f'This is comactness:{compactness_df}')
# print(f'This is assy:{assymetry_df}')

