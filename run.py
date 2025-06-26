#!/usr/bin/env python3
"""
Simplified run script for the Predictive Maintenance System
This script provides easy commands to run different parts of the system
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_setup():
    """Run the complete setup process"""
    logger.info("Running complete system setup...")
    try:
        subprocess.run([sys.executable, "setup.py"], check=True)
        logger.info("✓ Setup completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Setup failed: {e}")
        return False

def generate_data_only():
    """Generate only the sample data"""
    logger.info("Generating sample data...")
    try:
        from data_generator import generate_sensor_data, save_dataset
        df = generate_sensor_data(num_samples=10000, num_machines=50)
        save_dataset(df)
        logger.info("✓ Sample data generated successfully!")
        return True
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return False

def train_models_only():
    """Train only the ML models"""
    logger.info("Training ML models...")
    try:
        from ml_model import main
        main()
        logger.info("✓ Models trained successfully!")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_server(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI server"""
    logger.info(f"Starting server on {host}:{port}")
    
    # Check if models exist
    if not Path("xgb_model.pkl").exists() and not Path("rf_model.pkl").exists():
        logger.warning("No trained models found. Running setup first...")
        if not run_setup():
            return False
    
    try:
        cmd = [
            "uvicorn", "main:app", 
            "--host", host, 
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed to start: {e}")
        return False
    
    return True

def run_tests():
    """Run basic system tests"""
    logger.info("Running system tests...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check if data file exists
    total_tests += 1
    if Path("equipment_data.csv").exists():
        logger.info("✓ Test 1 passed: Data file exists")
        tests_passed += 1
    else:
        logger.error("✗ Test 1 failed: Data file missing")
    
    # Test 2: Check if models exist
    total_tests += 1
    if Path("xgb_model.pkl").exists() or Path("rf_model.pkl").exists():
        logger.info("✓ Test 2 passed: ML models exist")
        tests_passed += 1
    else:
        logger.error("✗ Test 2 failed: No trained models found")
    
    # Test 3: Check if database can be created
    total_tests += 1
    try:
        from database import init_database
        init_database()
        logger.info("✓ Test 3 passed: Database initialization works")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Test 3 failed: Database error - {e}")
    
    # Test 4: Test ML model loading
    total_tests += 1
    try:
        from ml_model import PredictiveMaintenanceModel
        model = PredictiveMaintenanceModel()
        if model.load_model('xgb_model.pkl') or model.load_model('rf_model.pkl'):
            logger.info("✓ Test 4 passed: ML model loads successfully")
            tests_passed += 1
        else:
            logger.error("✗ Test 4 failed: Cannot load ML models")
    except Exception as e:
        logger.error(f"✗ Test 4 failed: Model loading error - {e}")
    
    # Test 5: Test API imports
    total_tests += 1
    try:
        import main
        logger.info("✓ Test 5 passed: API module imports successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Test 5 failed: API import error - {e}")
    
    logger.info(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! System is ready to run.")
        return True
    else:
        logger.error("❌ Some tests failed. Please run setup or check the issues above.")
        return False

def show_status():
    """Show current system status"""
    logger.info("System Status Check")
    logger.info("=" * 50)
    
    # Check files
    files_to_check = {
        "equipment_data.csv": "Sample data file",
        "xgb_model.pkl": "XGBoost model",
        "rf_model.pkl": "Random Forest model", 
        "maintenance.db": "SQLite database",
        "static/index.html": "Dashboard frontend"
    }
    
    for file_path, description in files_to_check.items():
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            logger.info(f"✓ {description}: {file_path} ({size:,} bytes)")
        else:
            logger.info(f"✗ {description}: {file_path} (missing)")
    
    # Check directories
    directories = ["static", "models", "data", "logs"]
    logger.info("\nDirectories:")
    for directory in directories:
        if Path(directory).exists():
            logger.info(f"✓ {directory}/")
        else:
            logger.info(f"✗ {directory}/ (missing)")
    
    # Check Python packages
    logger.info("\nKey Dependencies:")
    packages_to_check = ["fastapi", "uvicorn", "sklearn", "xgboost", "pandas", "plotly"]
    
    for package in packages_to_check:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.info(f"✗ {package} (not installed)")

def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance System Runner")
    parser.add_argument("command", choices=[
        "setup", "data", "train", "serve", "test", "status", "docker"
    ], help="Command to run")
    
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        logger.info("🚀 Running complete system setup...")
        if run_setup():
            logger.info("🎉 Setup completed! You can now run: python run.py serve")
            return 0
        return 1
    
    elif args.command == "data":
        logger.info("📊 Generating sample data...")
        return 0 if generate_data_only() else 1
    
    elif args.command == "train":
        logger.info("🤖 Training ML models...")
        return 0 if train_models_only() else 1
    
    elif args.command == "serve":
        logger.info("🌐 Starting web server...")
        return 0 if run_server(args.host, args.port, args.reload) else 1
    
    elif args.command == "test":
        logger.info("🧪 Running system tests...")
        return 0 if run_tests() else 1
    
    elif args.command == "status":
        show_status()
        return 0
    
    elif args.command == "docker":
        logger.info("🐳 Building and running with Docker...")
        try:
            logger.info("Building Docker image...")
            subprocess.run(["docker", "build", "-t", "predictive-maintenance", "."], check=True)
            
            logger.info("Running Docker container...")
            subprocess.run([
                "docker", "run", "-p", f"{args.port}:8000", 
                "--name", "predictive-maintenance-app",
                "predictive-maintenance"
            ], check=True)
            
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker command failed: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)