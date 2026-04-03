from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ----------------------------
# Load model
# ----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ----------------------------
# Load scaler
# ----------------------------
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Vehicle Engine Failure Prediction API",
    version="1.0",
    description="Predicts probability of engine failure"
)

# ----------------------------
# Input validation 
# ----------------------------
class EngineData(BaseModel):
    temperature: float = Field(..., ge=250, le=400)
    rpm: float = Field(..., ge=0, le=4000)
    torque: float = Field(..., ge=0, le=200)
    tool_wear: float = Field(..., ge=0, le=300)

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_failure(data: EngineData):

    # correct feature order (must match training)
    features = np.array([[
        data.temperature,
        data.rpm,
        data.torque,
        data.tool_wear
    ]])

    # apply scaler if exists
    if scaler is not None:
        features = scaler.transform(features)

    # probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]
    else:
        prob = float(model.predict(features)[0])

    # CUSTOM THRESHOLD (IMPORTANT FIX)
    threshold = 0.3
    prediction = 1 if prob >= threshold else 0

    return {
        "prediction": prediction,
        "failure_probability": round(prob, 4),
        "risk_level": (
            "HIGH" if prob > 0.7 else
            "MEDIUM" if prob > 0.3 else
            "LOW"
        )
    }


@app.get("/")
def home():
    return {"status": "API is running"}
