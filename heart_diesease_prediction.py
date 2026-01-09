# Heart Disease Detection using Machine Learning
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 2. Load Dataset
data = pd.read_csv(r"C:\Users\1099TU\OneDrive\Documents\heart_disease\Heart Disease\dataset.csv")
print("First 5 rows of dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
# 3. Basic Data Exploration
print("\nTarget Value Counts:")
print(data['target'].value_counts())
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
# 4. Feature Selection
X = data.drop('target', axis=1)   # Features
y = data['target']                # Target
# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 7. Train Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# 8. Predictions
y_pred = model.predict(X_test)
# 9. Model Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# 10. Test with New Patient Data
# Example input: [age, sex, chest pain type, resting bp, cholesterol,
# fasting blood sugar, resting ecg, max heart rate, exercise angina, oldpeak, ST slope]
new_patient = np.array([[52, 1, 3, 130, 250, 0, 1, 150, 0, 1.0, 2]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
if prediction[0] == 1:
    print("\nPrediction: Patient has Heart Disease")
else:
    print("\nPrediction: Patient is Normal")
