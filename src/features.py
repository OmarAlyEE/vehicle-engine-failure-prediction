import pandas as pd
import os

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "ai4i2020.csv")

OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "features.csv")


# =========================
# LOAD
# =========================
df = pd.read_csv(DATA_PATH)

print("[OK] Loaded raw data:", df.shape)


# =========================
# VALIDATION CHECK
# =========================
required_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# =========================
# FEATURE ENGINEERING
# =========================

df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

df["wear_rate"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)

df["energy_proxy"] = (
    df["Torque [Nm]"] *
    df["Rotational speed [rpm]"] *
    df["Tool wear [min]"]
)

df["temp_ratio"] = df["Process temperature [K]"] / df["Air temperature [K]"]

# safety checks
assert df.isnull().sum().sum() == 0, "NaN detected after feature engineering"

print("[OK] Feature engineering completed")


# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Saved → {OUTPUT_PATH}")
print("Final shape:", df.shape)


# =========================
# FINAL CHECKS
# =========================
print("\n[CHECK] Class distribution:")
print(df["Machine failure"].value_counts(normalize=True))