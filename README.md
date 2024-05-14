# Skin Cancer Detection from Image Features - for Spring 2024 Projects in Data Science Course at ITU 

This repository contains Python scripts for processing medical images to extract features, training machine learning models to detect cancer based on these features, and predicting cancer likelihood with trained models.

## Project Structure

- `01_process_images.py`: Module to process images and extract features such as hue, saturation, and symmetry.
- `02_train_classifiers.py`: Module to train RandomForest models using extracted features, evaluate their performance, and save the best model.
- `03_evaluate_classifier.py`: Module to use a trained model to predict cancer likelihood and evaluate the model's accuracy.

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/niclejeune/projects-in-data-science.git
cd projects-in-data-science
```

### Requirements

This project requires Python 3.8 or later. Dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

### 0. Data Preparation

Download the necessary data files from [this Google Drive link](https://drive.google.com/file/d/1bZ0v7IBB9HPhRgtCViYxqghOzstXsoTp/view?usp=sharing) and extract them into the `data/` directory at the root of this project. Note: the `data/` directory is not included in this repository due to size constraints.


### 1. Feature Extraction (`01_process_images.py`)

This script processes images and metadata to extract relevant features for cancer detection. It accepts several optional command-line arguments:

- `--images_path`: Specifies the path to the images folder. 
  - Default: `data/images`
- `--masks_path`: Specifies the path to the masks folder. 
  - Default: `data/masks`
- `--metadata_path`: Specifies the path to the metadata CSV file. 
  - Default: `data/metadata.csv`
- `--output_csv`: Specifies the path where the output CSV file containing the extracted features will be saved. 
  - Default: `features/features.csv`

Example command to run:
```bash
python 01_process_images.py --images_path=data/images --masks_path=data/masks --metadata_path=data/metadata.csv --output_csv=features/features.csv
```

### 2. Model Training (`02_train_classifiers.py`)

This script trains models based on the extracted features and evaluates their performance. It also accepts several optional command-line arguments:

- `--features_path`: Specifies the path to the CSV file containing the extracted features data. 
  - Default: `features/features.csv`
- `--results_path`: Specifies the path where the training results will be saved or loaded from if the file already exists.
  - Default: `training/results.csv`
- `--graphs_path`: Specifies the path where graphs will be saved.
  - Default: `graphs/`
- `--models_path`: Specifies the directory path to save trained models if saving is desired.
  - Default: None (models are not saved unless this path is specified)
- `--final_model_path`: Specifies the path where the final model will be saved.
  - Default: `training/final_model`

Example command to run:
```bash
python 02_train_classifiers.py --features_path=features/features.csv --results_path=training/results.csv --graphs_path=graphs/ --models_path=training/models --final_model_path=training/final_model
```

### 3. Model Evaluation (`03_evaluate_classifier.py`)

This script uses a trained model to predict cancer likelihood and evaluates the model's accuracy. It accepts several optional command-line arguments:

- `--features_path`: Specifies the path to the features CSV file used for predictions.
  - Default: `features/features.csv`
- `--model_path`: Specifies the path to the trained model file.
  - Default: `training/final_model`
- `--results_path`: Specifies the path where the prediction results will be saved.
  - Default: `evaluation/results.csv`

Example command to run:
```bash
python 03_evaluate_classifier.py --features_path=features/features.csv --model_path=training/final_model --results_path=evaluation/results.csv
```

## Acknowledgments

Special thanks to ChatGPT for assistance in generating README content and crafting detailed docstrings for this project.

### Dataset Credit

This project uses the PAD-UFES-20 dataset, available at [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1). This dataset was collected by the Dermatological and Surgical Assistance Program at the Federal University of Espírito Santo (UFES), Brazil. It includes 2'298 samples of various skin lesions documented through clinical images and metadata.

Out of the 2'298 samples, 1'272 were used to train and test the model.

## License

The authors of this project reserve all rights.

