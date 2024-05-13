import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging

import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(description="Process image and mask paths for feature extraction.")
    parser.add_argument('--images_path', type=str, default='data/images', help='Path to the images folder')
    parser.add_argument('--masks_path', type=str, default='data/masks', help='Path to the masks folder')
    parser.add_argument('--metadata_path', type=str, default='data/metadata.csv', help='Path to the metadata csv file')
    parser.add_argument('--output_csv', type=str, default='features/features.csv',
                        help='Path to save the output CSV file')
    return parser.parse_args()


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load and preprocess the metadata from a CSV file.

    Args:
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        pd.DataFrame: Processed metadata with a new binary column indicating cancer presence.
    """
    metadata = pd.read_csv(metadata_path)
    metadata['cancer'] = metadata['diagnostic'].apply(lambda x: 1 if x in ['BCC', 'SCC', 'MEL'] else 0)
    return metadata


def process_images(images_path: str, masks_path: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Process each image and extract features using masks.

    Args:
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing corresponding masks.
        metadata (pd.DataFrame): Dataframe containing metadata information.

    Returns:
        pd.DataFrame: DataFrame containing extracted features along with diagnostic and cancer information.
    """
    features_list = []
    for image_filename in os.listdir(images_path):
        if image_filename.endswith(".png"):
            base_name = image_filename[:-4]
            image_path = os.path.join(images_path, image_filename)
            mask_path = os.path.join(masks_path, base_name + '_mask.png')

            image = plt.imread(image_path)[:, :, :3]
            mask = plt.imread(mask_path)

            features = extract_features.extract_features(image, mask)
            row = [image_filename] + metadata.loc[metadata['img_id'] == image_filename, ['diagnostic', 'cancer']].values.flatten().tolist() + features
            features_list.append(row)

    features_df = pd.DataFrame(features_list, columns=['img_id', 'diagnostic', 'cancer', 'hue', 'saturation', 'value',
                                                        'compactness_score', 'Vertical Asymmetry_mean',
                                                       'Horizontal Asymmetry_mean', 'dots'])
    return features_df


def main() -> None:
    """
    Main function to execute the process.
    """
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    features_df = process_images(args.images_path, args.masks_path, metadata)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    features_df.to_csv(args.output_csv, index=False)
    logging.info(f'Features saved to {args.output_csv}')


if __name__ == '__main__':
    main()
