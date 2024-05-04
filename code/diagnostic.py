import csv
import os

def read_csv_metadata(file_path):
    """Reads CSV metadata and returns a dictionary mapping img_id to diagnostic."""
    diagnostic_data = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_id = row['img_id'].replace('.png', '')  # Remove '.png' from img_id
            diagnostic_data[img_id] = row['diagnostic']
    return diagnostic_data

def get_diagnostic(file_path, diagnostic_data):
    """Extracts the file identifier and returns its diagnostic from the data dictionary."""
    file_id = file_path.split('/')[-1].replace('_mask.png', '')
    return diagnostic_data.get(file_id, "Diagnostic not found")

def save_to_csv(data, file_path):
    """Saves the given data to a CSV file."""
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'diagnostic'])  # header row
        for key, value in data.items():
            writer.writerow([key, value])

def process_directory(metadata_csv, directory, output_csv):
    # Read the CSV and store the data
    diagnostic_data = read_csv_metadata(metadata_csv)

    # Dictionary to store diagnostics extracted from files
    file_diagnostics = {}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            diagnostic_value = get_diagnostic(f, diagnostic_data)
            file_diagnostics[f] = diagnostic_value

    save_to_csv(file_diagnostics, output_csv)