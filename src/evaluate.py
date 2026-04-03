import os
import json
import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)

from sklearn.dummy import DummyClassifier

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "models", "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

target = "Machine failure"

# =========================
# CLEAN FEATURES (MATCH TRAINING)
# =========================
X = df.drop(columns=[target])
y = df[target]

leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
X = X.drop(columns=[c for c in leakage_cols if c in X.columns])
X = X.select_dtypes(include=["number"])

print("[OK] Feature shape:", X.shape)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)
print("[OK] Model loaded")

# =========================
# PREDICTIONS
# =========================
preds = model.predict(X)

probs = None
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)[:, 1]

# =========================
# THRESHOLD OPTIMIZATION (IMPORTANT)
# =========================
best_threshold = 0.5
best_f1 = 0

if probs is not None:
    thresholds = np.linspace(0.1, 0.9, 50)

    for t in thresholds:
        temp_preds = (probs >= t).astype(int)
        f1 = f1_score(y, temp_preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    preds = (probs >= best_threshold).astype(int)

print(f"[OK] Best threshold: {best_threshold:.3f}")

# =========================
# METRICS
# =========================
cm = confusion_matrix(y, preds)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(y, preds, zero_division=0)
recall = recall_score(y, preds, zero_division=0)
f1 = f1_score(y, preds, zero_division=0)

roc_auc = roc_auc_score(y, probs) if probs is not None else None

# PR-AUC (VERY IMPORTANT for imbalanced problems)
pr_auc = None
if probs is not None:
    p, r, _ = precision_recall_curve(y, probs)
    pr_auc = auc(r, p)

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# =========================
# BASELINE MODEL (VERY IMPORTANT FOR CV)
# =========================
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)
baseline_preds = dummy.predict(X)

baseline_f1 = f1_score(y, baseline_preds, zero_division=0)

# =========================
# PRINT RESULTS
# =========================
print("\n========================")
print("FINAL MODEL EVALUATION")
print("========================")

print("Confusion Matrix:")
print(cm)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}" if roc_auc else "ROC-AUC  : N/A")
print(f"PR-AUC   : {pr_auc:.4f}" if pr_auc else "PR-AUC   : N/A")
print(f"False Positive Rate: {fpr:.4f}")

print("\n========================")
print("BASELINE COMPARISON")
print("========================")
print(f"Baseline F1 (dummy): {baseline_f1:.4f}")
print(f"Model F1 improvement: {(f1 - baseline_f1):.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(y, preds, output_dict=True)

with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# =========================
# METRICS SAVE
# =========================
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc) if roc_auc else None,
    "pr_auc": float(pr_auc) if pr_auc else None,
    "false_positive_rate": float(fpr),
    "best_threshold": float(best_threshold),
    "confusion_matrix": cm.tolist()
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n[OK] Metrics saved")

# =========================
# CONFUSION MATRIX PLOT
# =========================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No Failure", "Failure"])
plt.yticks([0, 1], ["No Failure", "Failure"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# =========================
# FEATURE IMPORTANCE
# =========================
if hasattr(model, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    plt.figure()
    top = fi.head(10)[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")

    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))

# =========================
# BUSINESS INSIGHT
# =========================
print("\n========================")
print("BUSINESS INSIGHT")
print("========================")

print(f"False positives: {fp}")
print(f"False negatives: {fn}")

if fn > 0:
    print("Risk: missed failures (critical in predictive maintenance).")

if fp > 0:
    print("Warning: false alarms may increase maintenance cost.")

print("\n[OK] Evaluation complete")