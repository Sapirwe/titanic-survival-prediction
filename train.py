# train.py
import os
import random
import argparse
import numpy as np
import pandas as pd
import joblib
import subprocess
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Dataset
# -----------------------
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        # ColumnTransformer may return a sparse matrix; convert once to dense for PyTorch.
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        self.X = torch.tensor(X_dense, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------
# Model
# -----------------------
class TitanicModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_preprocessor(num_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()
        correct += (preds == y_batch.int()).sum().item()
        total += y_batch.size(0)

    return total_loss / len(loader), correct / total



def download_titanic(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)

    train_csv_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_csv_path):
        print("Titanic dataset already exists. Skipping download.")
        return train_csv_path

    # Check Kaggle credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "kaggle.json not found. Please place your Kaggle API token in ~/.kaggle/kaggle.json"
        )

    print("Downloading Titanic dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "titanic", "-p", data_dir],
        check=True,
    )

    zip_path = os.path.join(data_dir, "titanic.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    os.remove(zip_path)
    print("Download and extraction completed.")

    return train_csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="train.csv", help="Path to Kaggle Titanic train.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Features
    TARGET = "Survived"
    NUM_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    CAT_FEATURES = ["Sex", "Embarked"]
    DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

    # Load data
    if os.path.exists(args.csv_path):
        csv_path = args.csv_path
    else:
        csv_path = download_titanic()

    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv(os.path.join(args.artifacts_dir, "val.csv"), index=False)

    # Preprocess (fit only on train)
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # Save preprocessor for inference (Streamlit)
    joblib.dump(preprocessor, os.path.join(args.artifacts_dir, "preprocessor.pkl"))

    # PyTorch datasets/loaders
    train_dataset = TitanicDataset(X_train_proc, y_train)
    val_dataset = TitanicDataset(X_val_proc, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    input_dim = X_train_proc.shape[1]
    model = TitanicModel(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Train/Val sizes: {len(train_dataset)} / {len(val_dataset)}")
    print(f"Input dim: {input_dim}")

    # Train loop
    best_val_acc = 0.0
    best_path = os.path.join(args.artifacts_dir, "model.pt")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    print(f"Saved best model to: {best_path} (best Val Acc: {best_val_acc:.4f})")


if __name__ == "__main__":
    main()
