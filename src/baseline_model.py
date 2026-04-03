import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "features.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "baseline_results.csv")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("[OK] Loaded data:", df.shape)

target = "Machine failure"

X = df.drop(columns=[target])
y = df[target]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# LEARN SIMPLE RULE THRESHOLDS
# (data-driven baseline, not random)
# =========================

def compute_thresholds(X_train, y_train):
    failure_data = X_train[y_train == 1]

    thresholds = {
        "temp_diff": failure_data["temp_diff"].mean(),
        "power": failure_data["power"].mean(),
        "wear_rate": failure_data["wear_rate"].mean(),
        "energy_proxy": failure_data["energy_proxy"].mean()
    }

    print("\n[OK] Learned thresholds from failure cases:")
    for k, v in thresholds.items():
        print(f"{k}: {v:.4f}")

    return thresholds

thresholds = compute_thresholds(X_train, y_train)

# =========================
# RULE-BASED PREDICTION MODEL
# =========================
def predict(X, thresholds):
    preds = []

    for _, row in X.iterrows():

        score = 0

        if row["temp_diff"] > thresholds["temp_diff"]:
            score += 1
        if row["power"] > thresholds["power"]:
            score += 1
        if row["wear_rate"] > thresholds["wear_rate"]:
            score += 1
        if row["energy_proxy"] > thresholds["energy_proxy"]:
            score += 1

        # decision rule
        preds.append(1 if score >= 2 else 0)

    return np.array(preds)

# =========================
# PREDICT
# =========================
y_pred = predict(X_test, thresholds)

# =========================
# EVALUATION
# =========================
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=========================")
print("BASELINE MODEL RESULTS")
print("=========================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nConfusion Matrix:")
print(cm)

print(f"\nFalse Positives: {fp}")
print(f"False Negatives: {fn}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# SAVE RESULTS
# =========================
results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = y_pred

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(f"\n[OK] Saved baseline results → {OUTPUT_PATH}")