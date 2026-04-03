import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean.csv")


# =========================
# LOAD DATA
# =========================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at: {path}")

    df = pd.read_csv(path)
    print(f"[OK] Loaded data: {df.shape}")

    assert df.shape[0] > 0, "Empty dataset loaded"
    return df


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df.copy()

    # Drop useless identifiers
    df.drop(columns=[c for c in ["UDI", "Product ID"] if c in df.columns], inplace=True)

    # Encode categorical safely
    if "Type" in df.columns:
        df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

        if df["Type"].isnull().any():
            raise ValueError("Type column has unexpected values")

    print("[OK] Data cleaned")
    return df


# =========================
# MISSING VALUES
# =========================
def handle_missing(df):
    df = df.copy()

    missing_before = df.isnull().sum().sum()

    if missing_before > 0:
        df = df.fillna(df.median(numeric_only=True))

    missing_after = df.isnull().sum().sum()

    print(f"[OK] Missing values handled (before={missing_before}, after={missing_after})")

    assert missing_after == 0, "Missing values still exist!"
    return df


# =========================
# SPLIT DATA
# =========================
def split_data(df):
    if "Machine failure" not in df.columns:
        raise ValueError("Target column missing")

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    print("[OK] Splitting data...")

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# SCALE DATA
# =========================
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[OK] Scaling completed")

    # sanity check
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1]

    return X_train_scaled, X_test_scaled, scaler


# =========================
# SAVE
# =========================
def save_processed(X_train, X_test, y_train, y_test, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    train = pd.DataFrame(X_train)
    train["Machine failure"] = y_train.values

    test = pd.DataFrame(X_test)
    test["Machine failure"] = y_test.values

    full = pd.concat([train, test])

    # validation
    assert full.isnull().sum().sum() == 0, "NaN detected in final dataset"

    full.to_csv(path, index=False)
    print(f"[OK] Saved → {path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_data(DATA_PATH)

    df = clean_data(df)
    df = handle_missing(df)

    print("\nClass distribution:")
    print(df["Machine failure"].value_counts(normalize=True))

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_data(X_train, X_test)

    save_processed(X_train, X_test, y_train, y_test, OUTPUT_PATH)

    print("\n[PIPELINE OK] ETL completed successfully")