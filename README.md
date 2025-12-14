# titanic-survival-prediction

Overview
This project implements an end-to-end classification pipeline for predicting passenger survival on the Titanic dataset using PyTorch.
The solution includes data preprocessing, model training, and an interactive Streamlit app for inference and evaluation.

Only train.csv from Kaggle is used. The dataset is split into training and validation sets as part of the training process.

# Setup
# Requirements

Python 3.9+

Kaggle API credentials (kaggle.json)

Install dependencies:

pip install -r requirements.txt

# Training

To train the model, run:

python train.py


The training script:

Downloads the Titanic dataset from Kaggle

Splits train.csv into train/validation sets (80/20, stratified)

Applies preprocessing

Trains a PyTorch classification model

Saves the trained artifacts to disk

# Inference & Evaluation (Streamlit)

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

Supports adjusting the classification threshold interactively

The app can also be used to evaluate the held-out validation split saved during training.

# Artifacts

After training, the following files are created:

artifacts/model.pt – trained PyTorch model weights

artifacts/preprocessor.pkl – fitted preprocessing pipeline

These artifacts are loaded by the Streamlit app for inference and evaluation.

# Model & Preprocessing (Summary)

Numerical features: median imputation + standard scaling

Categorical features: most-frequent imputation + one-hot encoding

Model: feedforward neural network (2 fully connected layers)

Loss: Binary Cross Entropy with Logits

# Exploratory data analysis (EDA) is provided in eda.ipynb.
