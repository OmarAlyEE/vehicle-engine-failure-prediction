from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path

# ----------------------------
# Resolve project root safely
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ----------------------------
# Safety check (important)
# ----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}. "
        "Run src/save_model.py first."
    )

# ----------------------------
# Load model
# ----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Vehicle Engine Failure Prediction API",
    version="1.0",
    description="Predicts probability of engine failure from sensor data"
)

# ----------------------------
# Input schema
# ----------------------------
class EngineData(BaseModel):
    temperature: float
    rpm: float
    torque: float
    tool_wear: float

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_failure(data: EngineData):
    features = np.array([[
        data.temperature,
        data.rpm,
        data.torque,
        data.tool_wear
    ]])

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]
    else:
        prob = float(model.predict(features)[0])

    return {
        "failure_probability": round(prob, 4)
    }

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def home():
    return {"status": "API is running"}