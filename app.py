import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

MODEL_DIR = Path("model")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "model.joblib")
    features = json.loads((MODEL_DIR / "feature_names.json").read_text())
    return model, features

st.title("🩺 Breast Cancer Detection")
st.write("Educational demo — not for medical use.")

model, features = load_model()

# Input fields
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        inputs[feat] = st.number_input(feat, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])[features]
    proba = model.predict_proba(df)[:,1][0]
    pred = "Malignant" if proba >= 0.5 else "Benign"
    st.success(f"Prediction: {pred}")
    st.info(f"Probability malignant: {proba:.3f}")
