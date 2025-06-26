import joblib
import numpy as np

def load_model_and_scaler(model_path, scaler_path, features_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    return model, scaler, features

def predict_failure(model, scaler, features, data):
    input_data = np.array([[data.temperature, data.vibration, data.usage_hours, data.hour, data.day]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    return prob, prob > 0.7  # Threshold for maintenance