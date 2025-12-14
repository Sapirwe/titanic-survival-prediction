# Titanic Survival Prediction

## Overview
This project implements an end-to-end classification pipeline for predicting passenger survival on the Titanic dataset using **PyTorch**.  
The solution includes data preprocessing, model training, and an interactive **Streamlit** app for inference and evaluation.

Only `train.csv` from Kaggle is used. The dataset is split into training and validation sets as part of the training process.

---

## Setup

### Requirements
- Python 3.9+
- Kaggle API credentials (`kaggle.json`)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model, run:

python train.py


The training script:

Downloads the Titanic dataset from Kaggle

Splits train.csv into train/validation sets (80/20, stratified)

Applies preprocessing

Trains a PyTorch classification model

Saves the trained artifacts to disk

## Inference & Evaluation (Streamlit)

To launch the Streamlit app:

streamlit run ds_app.py


The app:

Loads the trained model and preprocessing pipeline from disk

Allows uploading a labeled CSV file (same schema as Kaggle train.csv)

Runs inference and displays:

Accuracy

Confusion Matrix

Predicted probability distribution

Sample prediction table

Supports interactive adjustment of the classification threshold

The app can also be used to evaluate the held-out validation split saved during training.

## Artifacts

After training, the following files are created:

artifacts/model.pt – trained PyTorch model weights

artifacts/preprocessor.pkl – fitted preprocessing pipeline

artifacts/val.csv – held-out validation split

These artifacts are loaded by the Streamlit app for inference and evaluation.
The uploaded CSV can be either the original train.csv or the validation split saved during training.

## Model & Preprocessing (Summary)

- Numerical features are processed using median imputation followed by standard scaling  
- Categorical features are processed using most-frequent imputation followed by one-hot encoding  
- The model is a feedforward neural network with two fully connected layers  
- The training objective uses Binary Cross Entropy with Logits loss  

## Exploratory Data Analysis

Exploratory data analysis (EDA) is provided in the Jupyter notebook `Titanic_eda.ipynb`.

## Example Usage

Below is an example of the Streamlit application after uploading a labeled CSV file.
<img width="1205" height="837" alt="image" src="https://github.com/user-attachments/assets/ef756cc1-84c7-49f5-949e-c7a38834d23d" />

