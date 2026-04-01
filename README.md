# vehicle-engine-failure-prediction
Predict whether a vehicle engine will fail soon based on sensor data.

📊 Dataset
Source

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
