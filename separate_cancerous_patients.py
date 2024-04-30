import csv
import shutil
import os

with  open ('./additional_files/image_ids_groups.csv', 'r') as file:
    csv_file = csv.reader(file)
    next(csv_file)
    for line in csv_file:
        if line[2] == "A":
            diagnostics = line[1]
            image_id = line[0]
            if diagnostics == "BCC" or diagnostics == "SCC" or diagnostics == "MEL":
                destination = './group_A_masks/Cancerous'
                shutil.move(f'./group_A_masks/{image_id}',destination)
            else:
                destination = './group_A_masks/Non-Cancerous'
                shutil.move(f'./group_A_masks/{image_id}', destination)
