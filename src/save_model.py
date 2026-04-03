import pickle
from pathlib import Path

MODEL_PATH = "models/model.pkl"


def save_model(model, path: str = MODEL_PATH):
    """
    Save trained ML model to disk using pickle.
    """
    Path("models").mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {path}")


if __name__ == "__main__":
    # Import trained model from training script
    from train import best_model

    save_model(best_model)