import os
import pandas as pd
import argparse
import logging
import extract_features
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for customizable image, mask, metadata paths, and output CSV path.

    Returns:
        argparse.Namespace: The namespace object containing all arguments.
    """
    parser = argparse.ArgumentParser(description="Process image and metadata paths for feature extraction.")
    parser.add_argument('--images_path', type=str, default='data/images', help='Path to the images folder')
    parser.add_argument('--masks_path', type=str, default='data/masks', help='Path to the masks folder')
    parser.add_argument('--metadata_path', type=str, default='data/metadata.csv', help='Path to the metadata csv')
    parser.add_argument('--output_csv', type=str, default='features/features.csv',
                        help='Path to save the output CSV file')
    return parser.parse_args()


def creating_basic_df(metadata_path: str) -> pd.DataFrame:
    """
    Prepares the main dataframe from the metadata CSV file.

    Args:
        metadata_path (str): The path to the metadata CSV file.

    Returns:
        pd.DataFrame: The dataframe with necessary diagnostic data and cancer flag.
    """
    meta_data = pd.read_csv(metadata_path)
    meta_data = meta_data[['img_id', 'diagnostic']]
    meta_data['cancer'] = 0  # Initialize cancer column with 0
    meta_data.loc[meta_data['diagnostic'].isin(['BCC', 'SCC', 'MEL']), 'cancer'] = 1  # Mark cancer cases
    return meta_data


def creating_missing_df_img(images_path: str, masks_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates dataframes for evenness, compactness, and asymmetry from the image and mask data.

    Args:
        images_path (str): Path to the images directory.
        masks_path (str): Path to the masks directory.

    Returns:
        tuple: A tuple containing dataframes for evenness, compactness, and asymmetry features.
    """
    logging.info('Processing evenness data...')
    evenness_score = extract_features.process_images_with_hsv_uniformity(images_path, masks_path)
    evenness_df = pd.DataFrame(evenness_score)

    logging.info('Processing compactness data...')
    compactness_score = extract_features.process_all_images(masks_path)
    compactness_df = pd.DataFrame(compactness_score)

    logging.info('Processing asymmetry data...')
    asymmetry_score = extract_features.main_assy_fun(masks_path)
    asymmetry_df = pd.DataFrame(asymmetry_score)

    return evenness_df, compactness_df, asymmetry_df


def merging_data_frames(meta_df: pd.DataFrame, even_df: pd.DataFrame, compact_df: pd.DataFrame, assy_df: pd.DataFrame,
                        output_csv: str) -> None:
    """
    Merges the metadata dataframe with evenness, compactness, and asymmetry dataframes and saves the result as a CSV.

    Args:
        meta_df (pd.DataFrame): Metadata dataframe.
        even_df (pd.DataFrame): Evenness dataframe.
        compact_df (pd.DataFrame): Compactness dataframe.
        assy_df (pd.DataFrame): Asymmetry dataframe.
        output_csv (str): Output CSV file path.
    """
    logging.info('Combining data frames...')
    combined_df = pd.merge(meta_df, even_df, on='img_id')
    combined_df_2 = pd.merge(combined_df, compact_df, on='img_id')
    combined_df_3 = pd.merge(combined_df_2, assy_df, on='img_id')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the directory exists
    combined_df_3.to_csv(output_csv, index=False)
    logging.info(f'CSV file saved at {output_csv}')


def main() -> None:
    """
    Main function to orchestrate the data parsing, processing, and saving.
    """
    args = parse_args()
    meta_data_df = creating_basic_df(args.metadata_path)
    evenness_df, compactness_df, asymmetry_df = creating_missing_df_img(args.images_path, args.masks_path)
    merging_data_frames(meta_data_df, evenness_df, compactness_df, asymmetry_df, args.output_csv)


if __name__ == '__main__':
    main()
