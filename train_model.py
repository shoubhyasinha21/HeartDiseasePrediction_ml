import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Create directory for the saved model
if not os.path.exists('model'):
    os.makedirs('model')

# 2. Load the heart disease dataset
df = pd.read_csv("dataset.csv")

# 3. Define features and target
# Based on your CSV: 'target' is 1 (disease) or 0 (normal)
X = df.drop('target', axis=1)
y = df['target']

# 4. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the model and the scaler
joblib.dump(model, "model/heart_model.pkl")
joblib.dump(scaler, "model/heart_scaler.pkl")

print("✅ Model and Scaler trained and saved successfully!")