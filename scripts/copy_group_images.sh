#!/bin/bash

# Path to the CSV file
csv_file="image_ids_groups.csv"

# Directory where images are stored
image_dir="images/"

# Directory to copy the images to (create if it doesn't exist)
destination_dir="group_A_images/"
mkdir -p "$destination_dir"

# Process the CSV file
while IFS=, read -r img_id diagnostic group
do
    # Check if the group is 'A' and if the image file exists
    if [[ $group == "A" ]] && [[ -f "$image_dir$img_id" ]]; then
        # Copy the image to the destination directory
        cp "$image_dir$img_id" "$destination_dir"
    fi
done < "$csv_file"
