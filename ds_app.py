import os
import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    
st.title("Titanic App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")

@st.cache_resource
def load_preprocessor(path: str):
    """Load the fitted preprocessing pipeline (cached for Streamlit reruns)."""
    return joblib.load(path)

@st.cache_resource
def load_model_and_dim(_preprocessor, model_path: str):
    """
    Load the trained model weights (state_dict) and reconstruct
    the PyTorch model architecture used during training.
    """
    state_dict = torch.load(model_path, map_location="cpu")
    input_dim = len(_preprocessor.get_feature_names_out())

    model = TitanicModel(input_dim)
    model.load_state_dict(state_dict)
    model.eval()

    return model, input_dim

# Ensure trained artifacts exist before running inference
if not os.path.exists(PREPROCESSOR_PATH):
    st.error(f"Missing preprocessor file: {PREPROCESSOR_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

preprocessor = load_preprocessor(PREPROCESSOR_PATH)
model, input_dim = load_model_and_dim(preprocessor, MODEL_PATH)

st.write("input_dim:", input_dim)
st.success("Model reconstructed from state_dict")

st.header("Upload CSV for evaluation")

uploaded_file = st.file_uploader(
    "Upload a CSV file (same schema as Kaggle train.csv, including 'Survived')",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw data preview")
    st.dataframe(df.head())

    # Classification threshold: affects predicted labels but not probabilities
    threshold = st.slider(
        "Classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    NUM_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    CAT_FEATURES = ["Sex", "Embarked"]
    DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

    # Drop columns that were removed during training to keep schema consistency
    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    required_cols = NUM_FEATURES + CAT_FEATURES + ["Survived"]
    missing_cols = [c for c in required_cols if c not in df_clean.columns]
    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        st.stop()

    # X/y exactly like in train.py
    X = df_clean[NUM_FEATURES + CAT_FEATURES]
    y = df_clean["Survived"]

    # Apply the fitted preprocessing pipeline (no refitting, no data leakage)
    X_proc = preprocessor.transform(X)

    # Convert preprocessed matrix to dense (if sparse) and then to torch tensor
    X_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int().numpy()

    y_true = y.astype(int).to_numpy()
    accuracy = (preds == y_true).mean()
    st.metric("Accuracy", f"{accuracy:.3f}")
    st.write("Predicted survivors:", int(preds.sum()), "out of", len(preds))

    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    # Confusion matrix to analyze types of classification errors
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"]
    ))

    results = pd.DataFrame({
        "prob_survived": probs.numpy(),
        "pred_survived": preds,
        "true_survived": y_true,
    })

    if "PassengerId" in df.columns:
        results.insert(0, "PassengerId", df["PassengerId"].values)

    st.subheader("Predictions preview")
    st.dataframe(results.head(20))

    # Visualize the distribution of predicted survival probabilities
    st.subheader("Predicted survival probability distribution")
    fig, ax = plt.subplots()
    ax.hist(probs.numpy(), bins=20)
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Number of passengers")
    st.pyplot(fig)


