import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Heart Disease Detector", page_icon="❤️")

# Load Assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "heart_scaler.pkl")

@st.cache_resource
def load_resources():
    if os.path.exists(model_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

model, scaler = load_resources()

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's clinical parameters to predict heart disease.")

if model is None:
    st.error("Model files not found! Please run 'python train_model.py' first.")
else:
    # Organize inputs into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 40)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting BP (mm Hg)", 50, 250, 120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", [1, 2, 3])

    # Predict Button
    if st.button("Predict Results", type="primary"):
        # Arrange inputs exactly like the training data
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
        
        # Scale and Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.error(f"### Result: Heart Disease Detected")
            st.write(f"Confidence Level: {probability:.2%}")
        else:
            st.success(f"### Result: Normal / No Disease")
            st.write(f"Confidence Level: {(1-probability):.2%}")