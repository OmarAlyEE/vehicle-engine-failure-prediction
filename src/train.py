import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
print("[OK] Loaded data:", df.shape)

target = "Machine failure"

# =========================
# REMOVE DATA LEAKAGE
# =========================
leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

existing_leakage = [col for col in leakage_cols if col in df.columns]

if existing_leakage:
    print("[WARNING] Dropping leakage columns:", existing_leakage)
    df = df.drop(columns=existing_leakage)

# =========================
# 🚨 USE SAME FEATURES AS API
# =========================
required_features = [
    "Air temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

missing = [col for col in required_features if col not in df.columns]
if missing:
    raise ValueError(f"Missing required features in dataset: {missing}")

X = df[required_features]
y = df[target]

print("[OK] Using features:", required_features)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("[OK] Train/Test split done")

# =========================
# MODEL (Random Forest ONLY)
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_test)

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
else:
    roc = None

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
cm = confusion_matrix(y_test, preds)

print("\n========================")
print("MODEL RESULTS")
print("========================")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc if roc else 'N/A'}")
print("\nConfusion Matrix:")
print(cm)

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)

print(f"\n[OK] Model saved to {model_path}")