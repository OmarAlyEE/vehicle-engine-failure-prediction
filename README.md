# Predictive Maintenance: Machine Failure Prediction (End-to-End MLOps System)

---

# Problem Statement

Industrial machines operate under continuous stress, and unexpected failures can cause significant downtime and financial loss.  
The goal of this project is to build a machine learning system that predicts machine failure using sensor data, and deploy it as a production-ready API with monitoring and CI/CD automation.

---

# Dataset

This project uses the:

AI4I 2020 Predictive Maintenance Dataset  
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset  

---

## Dataset Description

- 10,000 machine records  
- Sensor + operational data  
- Simulated industrial predictive maintenance scenario  

---

## Features

- Air temperature (K)  
- Process temperature (K)  
- Rotational speed (rpm)  
- Torque (Nm)  
- Tool wear (min)  
- Type (L, M, H)  

---

## Target Variable

- Machine failure (binary)  
  - 1 → failure  
  - 0 → normal operation  

---

## Failure Modes (removed to prevent leakage)

- TWF → Tool Wear Failure  
- HDF → Heat Dissipation Failure  
- PWF → Power Failure  
- OSF → Overstrain Failure  
- RNF → Random Failure  

---

## Data Characteristics

- 10,000 samples  
- No missing values  
- Class imbalance (rare failures)  

---

# System Architecture

Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Model Saving → FastAPI Deployment → Drift Monitoring → CI/CD Pipeline  

---

# Pipeline Overview

---

## 1. Data Preprocessing

- Removed identifiers (Product ID, Type encoding handling)  
- Removed leakage features (TWF, HDF, PWF, OSF, RNF)  
- Feature scaling using StandardScaler  
- Train/test split  

---

## 2. Feature Engineering

Engineered operational signals:

- Temperature difference  
- Power estimation  
- Wear rate  
- Energy proxy  
- Temperature ratio  

---

## 3. Baseline Model (Rule-Based System)

A threshold-based heuristic model using engineered features.

### Performance:
- Precision: 0.0579  
- Recall: 0.5735  

Demonstrates limitations of non-ML approaches  

---

## 4. Machine Learning Models

### Random Forest (Best Model)

- Accuracy: 0.9885  
- Precision: 0.8814  
- Recall: 0.7647  
- F1-score: 0.8189  
- ROC-AUC: 0.9871  

---

### XGBoost

- Accuracy: 0.9885  
- Precision: 0.8814  
- Recall: 0.7647  
- ROC-AUC: 0.9816  

---

### Isolation Forest (Anomaly Detection)

- Precision: 0.3469  
- Recall: 0.2500  
- F1-score: 0.2906  

---

# Final Model Selection

Random Forest was selected due to:

- Best balance between precision and recall  
- High ROC-AUC score  
- Stability across evaluation runs  
- Better interpretability than boosting models  

---

# Model Artifacts

models/best_model.pkl  

---

# FastAPI Deployment

A REST API was built to serve real-time predictions.

## Features

- JSON input validation  
- Risk scoring (failure probability)  
- Risk classification (LOW / MEDIUM / HIGH)  
- Real-time inference endpoint  

---

## Example Input

{
  "temperature": 320,
  "rpm": 3000,
  "torque": 100,
  "tool_wear": 250
}

---

## Output

{
  "prediction": 1,
  "failure_probability": 0.4691,
  "risk_level": "MEDIUM"
}

---

# Monitoring (MLOps Core)

Implemented a data drift detection system:

- Compares training vs new data distributions  
- Uses statistical shift scoring  
- Automatically logs drift reports  

### Output File
- monitoring/drift_report.json  

---

# CI/CD Pipeline (GitHub Actions)

Automated pipeline includes:

- Code checkout  
- Python environment setup  
- Dependency installation  
- Drift sanity check  
- Docker build validation  

### Triggered on:
- Push to main  
- Pull requests  

---

# Docker Support

- FastAPI service runs inside Docker  
- Ensures environment consistency  
- Enables reproducibility across environments  

---

# Key Results Summary

| Model | Precision | Recall | F1 | ROC-AUC |
|------|----------|--------|----|--------|
| Baseline | 0.06 | 0.57 | 0.11 | - |
| Random Forest | 0.88 | 0.76 | 0.82 | 0.987 |
| XGBoost | 0.88 | 0.76 | 0.82 | 0.982 |
| Isolation Forest | 0.35 | 0.25 | 0.29 | - |

---

# Key Insights

- Leakage features significantly inflated naive models  
- Tree-based models performed best  
- Feature engineering improved performance significantly  
- Production ML systems require drift monitoring  
- CI/CD is required for reproducibility  

---

# Technologies Used

- Python  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- FastAPI  
- Docker  
- GitHub Actions  
- Matplotlib / Seaborn  

---

# Limitations

- Dataset is a benchmark dataset  
- Cleaner than real industrial environments  
- Less noise and simpler failure patterns  

Best suited for ML/MLOps pipeline validation rather than fully realistic production deployment  

---

# Future Improvements

- Hyperparameter tuning (Optuna / GridSearch)  
- SMOTE for class imbalance  
- Real-time streaming inference  
