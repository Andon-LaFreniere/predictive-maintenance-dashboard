#!/usr/bin/env python3
"""
Setup script for Predictive Maintenance System
This script sets up the entire system: generates data, trains models, and prepares the database
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr.strip()}")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'sqlalchemy', 'plotly', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.info("Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing dependencies"):
            logger.error("Failed to install dependencies. Please install them manually.")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    logger.info("Creating project directories...")
    
    directories = ['static', 'models', 'data', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True

def generate_data():
    """Generate sample IoT sensor data"""
    logger.info("Generating sample sensor data...")
    
    try:
        from data_generator import generate_sensor_data, save_dataset
        df = generate_sensor_data(num_samples=10000, num_machines=50)
        save_dataset(df, 'equipment_data.csv')
        logger.info("✓ Sample data generated successfully")
        return True
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        return False

def train_models():
    """Train machine learning models"""
    logger.info("Training machine learning models...")
    
    try:
        from ml_model import PredictiveMaintenanceModel
        
        # Initialize model
        pm_model = PredictiveMaintenanceModel()
        
        # Load and prepare data
        df = pm_model.load_and_preprocess_data('equipment_data.csv')
        X, y = pm_model.prepare_features(df)
        
        logger.info(f"Dataset shape: {X.shape}, Failure rate: {y.mean():.2%}")
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        X_test_rf, y_test_rf, y_pred_rf, y_pred_proba_rf = pm_model.train_random_forest(X, y)
        pm_model.evaluate_model(X_test_rf, y_test_rf, y_pred_rf, y_pred_proba_rf)
        pm_model.save_model('rf_model.pkl')
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        pm_model_xgb = PredictiveMaintenanceModel()
        df_xgb = pm_model_xgb.load_and_preprocess_data('equipment_data.csv')
        X_xgb, y_xgb = pm_model_xgb.prepare_features(df_xgb)
        X_test_xgb, y_test_xgb, y_pred_xgb, y_pred_proba_xgb = pm_model_xgb.train_xgboost(X_xgb, y_xgb)
        pm_model_xgb.evaluate_model(X_test_xgb, y_test_xgb, y_pred_xgb, y_pred_proba_xgb)
        pm_model_xgb.save_model('xgb_model.pkl')
        
        logger.info("✓ Models trained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return False

def setup_database():
    """Initialize database with sample data"""
    logger.info("Setting up database...")
    
    try:
        from database import init_database, populate_sample_equipment, load_sensor_data_to_db
        
        init_database()
        populate_sample_equipment()
        load_sensor_data_to_db('equipment_data.csv')
        
        logger.info("✓ Database setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

def verify_setup():
    """Verify that everything is set up correctly"""
    logger.info("Verifying setup...")
    
    # Check if files exist
    required_files = [
        'equipment_data.csv',
        'rf_model.pkl',
        'xgb_model.pkl',
        'maintenance.db',
        'static/index.html'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path} exists")
        else:
            logger.error(f"✗ {file_path} is missing")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("PREDICTIVE MAINTENANCE SYSTEM SETUP")
    logger.info("="*60)
    
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Creating directories", create_directories),
        ("Generating sample data", generate_data),
        ("Training ML models", train_models),
        ("Setting up database", setup_database),
        ("Verifying setup", verify_setup)
    ]
    
    for step_name, step_function in steps:
        logger.info(f"\nStep: {step_name}")
        logger.info("-" * 40)
        
        if not step_function():
            logger.error(f"Setup failed at step: {step_name}")
            sys.exit(1)
        
        logger.info(f"✓ {step_name} completed successfully")
    
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\nTo start the application:")
    logger.info("python main.py")
    logger.info("\nOr:")
    logger.info("uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("\nThe dashboard will be available at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()