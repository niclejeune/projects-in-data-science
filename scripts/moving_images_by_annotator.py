import csv
import shutil

with  open ('group01_imageids.csv', 'r') as file:
    csv_file = csv.reader(file)
    next(csv_file)
    for line in csv_file:
        annotator_id = line[1]
        image_id = line[0]
        # Copies from source to destination
        shutil.move(f'./group_A_images/{image_id}',f'./group_A_images/{annotator_id}/data/raw_data')
        #Folder structure is based on convert_json_to_image.py description