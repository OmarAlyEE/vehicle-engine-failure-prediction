import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib

from xgboost import XGBClassifier


# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
print("[OK] Loaded data:", df.shape)

target = "Machine failure"

# =========================
# 🚨 REMOVE DATA LEAKAGE
# =========================
leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

existing_leakage = [col for col in leakage_cols if col in df.columns]

if existing_leakage:
    print("[WARNING] Dropping leakage columns:", existing_leakage)
    df = df.drop(columns=existing_leakage)

# =========================
# SPLIT FEATURES / TARGET
# =========================
X = df.drop(columns=[target])
y = df[target]

# =========================
# REMOVE NON-NUMERIC COLUMNS
# =========================
non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

if non_numeric_cols:
    print("[WARNING] Dropping non-numeric columns:", non_numeric_cols)
    X = X.select_dtypes(include=["number"])

# safety check
assert X.select_dtypes(exclude=["number"]).shape[1] == 0, "Non-numeric columns still exist!"

print("[OK] Final feature count:", X.shape[1])

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
# SCALING (ONLY FOR XGBOOST)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# sanity checks
assert not np.isnan(X_train_scaled).any(), "NaNs in training data"
assert not np.isnan(X_test_scaled).any(), "NaNs in test data"

print("[OK] Scaling completed")

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(name, model, X_test, y_test):
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
    print(f"{name} RESULTS")
    print("========================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc if roc else 'N/A'}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc if roc else 0
    }


# =========================
# 1. RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
rf_results = evaluate_model("Random Forest", rf, X_test, y_test)


# =========================
# 2. XGBOOST
# =========================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train_scaled, y_train)
xgb_results = evaluate_model("XGBoost", xgb, X_test_scaled, y_test)


# =========================
# 3. ISOLATION FOREST
# =========================
iso = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

iso.fit(X_train)

iso_preds = iso.predict(X_test)
iso_preds = np.where(iso_preds == -1, 1, 0)

print("\n========================")
print("Isolation Forest RESULTS")
print("========================")

iso_prec = precision_score(y_test, iso_preds, zero_division=0)
iso_rec = recall_score(y_test, iso_preds, zero_division=0)
iso_f1 = f1_score(y_test, iso_preds, zero_division=0)

print(f"Precision: {iso_prec:.4f}")
print(f"Recall   : {iso_rec:.4f}")
print(f"F1-score : {iso_f1:.4f}")

iso_results = {
    "model": "Isolation Forest",
    "accuracy": 0,
    "precision": iso_prec,
    "recall": iso_rec,
    "f1": iso_f1,
    "roc_auc": 0
}

# =========================
# MODEL SELECTION
# =========================
results = [rf_results, xgb_results, iso_results]

best_model = max(results, key=lambda x: (x["f1"], x["recall"]))

print("\n========================")
print("BEST MODEL SELECTED")
print("========================")
print(best_model)

# =========================
# SAVE BEST MODEL
# =========================
if best_model["model"] == "Random Forest":
    joblib.dump(rf, os.path.join(MODEL_PATH, "best_model.pkl"))

elif best_model["model"] == "XGBoost":
    joblib.dump(xgb, os.path.join(MODEL_PATH, "best_model.pkl"))

print("\n[OK] Best model saved to models/best_model.pkl")