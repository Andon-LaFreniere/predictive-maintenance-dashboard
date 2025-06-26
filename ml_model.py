import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_type = None
        
    def load_and_preprocess_data(self, data_path='equipment_data.csv'):
        """Load and preprocess the equipment data"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Feature engineering
            df = self._engineer_features(df)
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _engineer_features(self, df):
        """Create additional features for better prediction"""
        # Temperature anomaly detection
        df['temp_anomaly'] = (df['temperature'] > df['temperature'].quantile(0.95)).astype(int)
        df['vibration_anomaly'] = (df['vibration'] > df['vibration'].quantile(0.95)).astype(int)
        
        # Interaction features
        df['temp_vibration_ratio'] = df['temperature'] / (df['vibration'] + 1e-8)
        df['pressure_speed_ratio'] = df['pressure'] / (df['rotation_speed'] + 1e-8)
        
        # Age-based features
        df['hours_per_maintenance'] = df['operating_hours'] / (df['maintenance_count'] + 1)
        
        # Operational efficiency metrics
        df['efficiency_score'] = (df['rotation_speed'] * df['pressure']) / (df['temperature'] + df['vibration'])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Encode categorical variables
        df_encoded = df.copy()
        df_encoded['machine_type_encoded'] = self.label_encoder.fit_transform(df['machine_type'])
        
        # Select features for training
        feature_columns = [
            'temperature', 'vibration', 'pressure', 'rotation_speed', 'current',
            'operating_hours', 'maintenance_count', 'hour', 'day_of_week', 'month',
            'temp_rolling_avg', 'vibration_rolling_avg', 'machine_type_encoded',
            'temp_anomaly', 'vibration_anomaly', 'temp_vibration_ratio',
            'pressure_speed_ratio', 'hours_per_maintenance', 'efficiency_score'
        ]
        
        self.feature_columns = feature_columns
        X = df_encoded[feature_columns]
        y = df_encoded['failure']
        
        return X, y
    
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest with hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        
        self.model = rf_grid.best_estimator_
        self.model_type = 'RandomForest'
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        logger.info(f"Best parameters: {rf_grid.best_params_}")
        logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def train_xgboost(self, X, y):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        xgb_params = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 3, 5]  # Handle class imbalance
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1)
        xgb_grid.fit(X_train_scaled, y_train)
        
        self.model = xgb_grid.best_estimator_
        self.model_type = 'XGBoost'
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        logger.info(f"Best parameters: {xgb_grid.best_params_}")
        logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def evaluate_model(self, X_test, y_test, y_pred, y_pred_proba):
        """Evaluate and visualize model performance"""
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(feature_importance.head(10))
            
            return feature_importance
        
        return None
    
    def predict_failure(self, equipment_data):
        """Predict failure probability for new equipment data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Prepare features
        features = equipment_data[self.feature_columns]
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        failure_probability = self.model.predict_proba(features_scaled)[:, 1]
        predictions = self.model.predict(features_scaled)
        
        return predictions, failure_probability
    
    def save_model(self, model_path='predictive_maintenance_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path='predictive_maintenance_model.pkl'):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model type: {self.model_type}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def main():
    """Main training pipeline"""
    # Initialize model
    pm_model = PredictiveMaintenanceModel()
    
    # Load and preprocess data
    df = pm_model.load_and_preprocess_data()
    X, y = pm_model.prepare_features(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Failure rate: {y.mean():.2%}")
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    X_test_rf, y_test_rf, y_pred_rf, y_pred_proba_rf = pm_model.train_random_forest(X, y)
    feature_importance_rf = pm_model.evaluate_model(X_test_rf, y_test_rf, y_pred_rf, y_pred_proba_rf)
    pm_model.save_model('rf_model.pkl')
    
    print("\n" + "="*50)
    print("TRAINING XGBOOST")
    print("="*50)
    pm_model_xgb = PredictiveMaintenanceModel()
    df_xgb = pm_model_xgb.load_and_preprocess_data()
    X_xgb, y_xgb = pm_model_xgb.prepare_features(df_xgb)
    X_test_xgb, y_test_xgb, y_pred_xgb, y_pred_proba_xgb = pm_model_xgb.train_xgboost(X_xgb, y_xgb)
    feature_importance_xgb = pm_model_xgb.evaluate_model(X_test_xgb, y_test_xgb, y_pred_xgb, y_pred_proba_xgb)
    pm_model_xgb.save_model('xgb_model.pkl')

if __name__ == "__main__":
    main()