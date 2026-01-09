from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)
CORS(app)
data = pd.read_csv("C:\\Users\\1099TU\\OneDrive\\Documents\\heart_disease\\Heart Disease\\dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

@app.route("/")
def home():
    return "Heart Disease Prediction API is Running"
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    input_data = np.array([data])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Patient has Heart Disease" if prediction[0] == 1 else "Patient is Normal"
    return jsonify({"prediction": result})
if __name__ == "__main__":
    app.run(debug=True)
