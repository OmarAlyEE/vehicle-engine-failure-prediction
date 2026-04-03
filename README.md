📌 Predictive Maintenance: Machine Failure Prediction
📖 Project Overview

This project builds an end-to-end machine learning system to predict equipment failures using sensor and operational data. The goal is to improve early detection of failures and reduce downtime in industrial systems.

The pipeline includes:

Data preprocessing and feature engineering
Baseline rule-based model
Machine learning models (Random Forest, XGBoost, Isolation Forest)
Model evaluation and comparison
Final model selection and saving


The dataset used in this project is the
AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository

🔗 https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Description

This dataset simulates a predictive maintenance scenario for industrial machines and contains 10,000 data points with multiple sensor readings.

Each row represents a machine observation with associated operational and environmental parameters.

Features

The dataset includes the following key features:

Air temperature 
𝐾
K
Process temperature 
𝐾
K
Rotational speed 
𝑟
𝑝
𝑚
rpm
Torque 
𝑁
𝑚
Nm
Tool wear 
𝑚
𝑖
𝑛
min
Type (L, M, H → machine quality category)
Target Variable
Machine failure (binary):
1 → failure occurred
0 → no failure
Additional Labels (Failure Types)

The dataset also includes detailed failure modes:

TWF → Tool Wear Failure
HDF → Heat Dissipation Failure
PWF → Power Failure
OSF → Overstrain Failure
RNF → Random Failure

💡 These are not used as primary targets in this project but can be leveraged for advanced modeling.

Data Characteristics
Total samples: 10,000
No missing values
Mixed data types (numerical + categorical)
Imbalanced target variable (few failure cases)


⚙️ Pipeline Stages
1. Data Preprocessing
Removed non-numeric identifiers (Product ID, Type)
Removed leakage columns (TWF, HDF, PWF, OSF, RNF)
Scaled numerical features using StandardScaler
Train/test split applied
2. Baseline Model (Rule-Based)

A simple threshold-based system was implemented using engineered features such as:

temperature difference
power
wear rate

Baseline performance:

Precision: 0.0579
Recall: 0.5735

This demonstrated the limitations of non-ML approaches.

3. Machine Learning Models

The following models were trained and evaluated:

Random Forest
Accuracy: 0.9885
Precision: 0.8814
Recall: 0.7647
F1-score: 0.8189
ROC-AUC: 0.9871
XGBoost
Accuracy: 0.9885
Precision: 0.8814
Recall: 0.7647
ROC-AUC: 0.9816
Isolation Forest (Anomaly Detection)
Precision: 0.3469
Recall: 0.2500
F1-score: 0.2906
🏆 Final Model Selection

The Random Forest model was selected as the best model due to:

Strong balance between precision and recall
High ROC-AUC score
Better interpretability compared to XGBoost
Stable performance across evaluation metrics

Final saved model:

models/best_model.pkl
📈 Key Results Summary
Model	Precision	Recall	F1-score	ROC-AUC
Baseline	0.06	0.57	0.11	-
Random Forest	0.88	0.76	0.82	0.987
XGBoost	0.88	0.76	0.82	0.982
Isolation Forest	0.35	0.25	0.29	-
🧠 Key Insights
Removing leakage features significantly improved model realism
Tree-based models performed best on this dataset
Baseline rule-based systems are insufficient for production use
Trade-off exists between false positives and missed failures
🚀 Technologies Used
Python
Pandas, NumPy
Scikit-learn
XGBoost
Matplotlib / Seaborn
📌 Future Improvements
Hyperparameter tuning (GridSearch / Optuna)
SMOTE for class imbalance handling
Feature importance analysis
Deployment using FastAPI / Docker
Real-time monitoring system
📂 Project Structure
src/
  baseline_model.py
  train.py
  preprocess.py
models/
data/
📊 Conclusion

This project demonstrates a full machine learning pipeline for predictive maintenance, achieving strong performance after addressing data leakage and feature selection issues.

📌 Dataset Limitations

The AI4I 2020 dataset is a well-structured benchmark dataset designed for predictive maintenance research.
It exhibits relatively clear class separability, which results in high model performance metrics (ROC-AUC ~0.99).

This makes it suitable for validating ML pipelines, feature engineering, and MLOps workflows, but not fully representative of noisy industrial environments where sensor drift, missing data, and delayed labeling are common.

Built and deployed a FastAPI-based machine learning service for real-time vehicle engine failure prediction, integrating a trained Random Forest model with structured input validation and probabilistic risk scoring.
