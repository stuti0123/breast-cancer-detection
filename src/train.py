import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATA_PATH = Path("data/breast_cancer.csv")
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    # last column assumed target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline with scaling + LR
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"ROC AUC: {roc_auc:.3f}")

    # Save artifacts
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

if __name__ == "__main__":
    main()
