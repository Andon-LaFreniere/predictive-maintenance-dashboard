import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data

def train_model(data_path, model_path):
    # Load and preprocess data
    X, y, scaler, features = load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(features, 'models/features.pkl')

if __name__ == "__main__":
    train_model('data/sensor_data.csv', 'models/xgboost_model.pkl')