"""
Merge n folders of masks and compute the area features of each mask to create a csv
Usage: python merge_masks.py PATH_MASK_FOLDERS
Example: python merge_masks.py ./masks

masks is a folder with the following structure
.
└── masks/
    ├── StudentA/
    │   ├── img1_mask.png
    │   └── img2_mask.png
    ├── StudentB/
    │   ├── img2_mask.png
    │   ├── img3_mask.png
    │   └── img4_mask.png
    └── StudentC/
        ├── img5_mask.png
        └── img6_mask.png
The generated csv will be created inside masks folder and contain the value for the area for each mask and each annotator

Example of csv:
img_name,feature_StudentA,feature_StudentB,feature_StudentC
PAT_8_15_820,30375.467,None,30375.467
PAT_39_55_233,None,142723.34,142723.34
"""

import os
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt

def area(mask):
    """
    Compute the area of an annotated lesion. Sum of pixels equal to 1
    """
    return np.sum(mask)

if __name__ == "__main__":
    #Get the path to mask folders' parent from the python command line
    path_masks_folders = sys.argv[1]
        
    #dictionary with image name as key and list of pair of annotator-feature as value. Will be saved into a CSV.
    dict_feature = {}

    #Get every subfolder containing masks
    masks_folders = next(os.walk(path_masks_folders))[1]

    for folder in masks_folders:
        #Get every mask in the folder
        masks = os.path.join(path_masks_folders,folder,"*.png")
        for mask_path in glob.glob(masks):
            img_name = mask_path.split("/")[-1].removesuffix("_mask.png")
            #Load the image and compute the area
            mask_img = plt.imread(mask_path)
            area_mask = area(mask_img)

            #Add the result into the dict
            if img_name not in dict_feature:
                dict_feature[img_name] = {name:None for name in masks_folders}
            dict_feature[img_name][folder] = area_mask
    
    #Create the CSV
    with open(os.path.join(path_masks_folders,"merged_annotations.csv"),"w") as csv_file:
        #Write the first line (columns' name)
        columns="img_name"
        for name in masks_folders:
            columns += f",feature_{name}"
        columns += "\n"
        csv_file.write(columns)

        #Write the data, 1 row per image
        for img in dict_feature:
            line = f"{img}"
            for name in masks_folders:
                line += f",{str(dict_feature[img][name])}" 
            line += "\n"
            csv_file.write(line)
    