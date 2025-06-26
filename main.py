from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
import logging
from pathlib import Path

# Import our modules
from ml_model import PredictiveMaintenanceModel
from database import (
    get_db, init_database, populate_sample_equipment, load_sensor_data_to_db,
    get_equipment_list, get_sensor_data, save_prediction, get_predictions,
    create_maintenance_schedule, get_maintenance_schedule, Equipment, SensorData
)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance System",
    description="ML-powered predictive maintenance system for industrial equipment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
ml_model = PredictiveMaintenanceModel()

# Pydantic models for API
class SensorDataInput(BaseModel):
    machine_id: str
    temperature: float
    vibration: float
    pressure: float
    rotation_speed: float
    current: float
    operating_hours: float
    maintenance_count: int = 0

class PredictionResponse(BaseModel):
    machine_id: str
    failure_probability: float
    predicted_failure: bool
    recommendation: str
    confidence_score: Optional[float] = None
    timestamp: datetime

class MaintenanceScheduleInput(BaseModel):
    machine_id: str
    scheduled_date: datetime
    maintenance_type: str
    priority: str = "medium"
    assigned_technician: Optional[str] = None
    estimated_duration: float = 4.0
    cost_estimate: float = 500.0
    notes: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize database and load ML model on startup"""
    logger.info("Starting up Predictive Maintenance System...")
    
    # Initialize database
    init_database()
    populate_sample_equipment()
    
    # Try to load sensor data
    try:
        load_sensor_data_to_db()
    except Exception as e:
        logger.warning(f"Could not load sensor data: {e}")
    
    # Load ML model
    if not ml_model.load_model('xgb_model.pkl'):
        logger.warning("Could not load XGBoost model, trying Random Forest...")
        if not ml_model.load_model('rf_model.pkl'):
            logger.error("No trained models found! Please train models first.")
    
    logger.info("System startup completed!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Predictive Maintenance System</title></head>
            <body>
                <h1>Predictive Maintenance System API</h1>
                <p>Welcome to the Predictive Maintenance System API!</p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/equipment">Equipment List</a></li>
                    <li><a href="/dashboard">Dashboard Data</a></li>
                </ul>
            </body>
        </html>
        """)
    except UnicodeDecodeError:
        logger.error("Unicode decode error reading index.html")
        return HTMLResponse(content="""
        <html>
            <head><title>Predictive Maintenance System</title></head>
            <body>
                <h1>Dashboard Loading Error</h1>
                <p>There was an encoding issue with the dashboard file.</p>
                <p>Try accessing the API directly:</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/dashboard">Dashboard Data (JSON)</a></li>
                    <li><a href="/equipment">Equipment List</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "model_loaded": ml_model.model is not None
    }

@app.get("/equipment")
async def get_equipment():
    """Get list of all equipment"""
    try:
        equipment = get_equipment_list()
        return [
            {
                "id": eq.id,
                "machine_id": eq.machine_id,
                "machine_type": eq.machine_type,
                "location": eq.location,
                "status": eq.status,
                "last_maintenance": eq.last_maintenance,
                "installation_date": eq.installation_date
            }
            for eq in equipment
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/equipment/{machine_id}/sensor-data")
async def get_equipment_sensor_data(machine_id: str, limit: int = 100):
    """Get recent sensor data for a specific machine"""
    try:
        sensor_data = get_sensor_data(machine_id, limit)
        return [
            {
                "id": data.id,
                "machine_id": data.machine_id,
                "timestamp": data.timestamp,
                "temperature": data.temperature,
                "vibration": data.vibration,
                "pressure": data.pressure,
                "rotation_speed": data.rotation_speed,
                "current": data.current,
                "operating_hours": data.operating_hours
            }
            for data in sensor_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_maintenance(sensor_input: SensorDataInput, background_tasks: BackgroundTasks):
    """Predict maintenance needs for equipment"""
    try:
        if ml_model.model is None:
            raise HTTPException(status_code=503, detail="ML model not loaded")
        
        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'temperature': sensor_input.temperature,
            'vibration': sensor_input.vibration,
            'pressure': sensor_input.pressure,
            'rotation_speed': sensor_input.rotation_speed,
            'current': sensor_input.current,
            'operating_hours': sensor_input.operating_hours,
            'maintenance_count': sensor_input.maintenance_count,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'temp_rolling_avg': sensor_input.temperature,
            'vibration_rolling_avg': sensor_input.vibration,
            'machine_type_encoded': 0,  # Default encoding
            'temp_anomaly': 1 if sensor_input.temperature > 80 else 0,
            'vibration_anomaly': 1 if sensor_input.vibration > 5 else 0,
            'temp_vibration_ratio': sensor_input.temperature / (sensor_input.vibration + 1e-8),
            'pressure_speed_ratio': sensor_input.pressure / (sensor_input.rotation_speed + 1e-8),
            'hours_per_maintenance': sensor_input.operating_hours / (sensor_input.maintenance_count + 1),
            'efficiency_score': (sensor_input.rotation_speed * sensor_input.pressure) / 
                              (sensor_input.temperature + sensor_input.vibration)
        }])
        
        # Make prediction
        predictions, probabilities = ml_model.predict_failure(input_data)
        
        failure_probability = float(probabilities[0])
        predicted_failure = bool(predictions[0])
        
        # Generate recommendation
        if failure_probability > 0.8:
            recommendation = "Immediate maintenance required - High failure risk"
        elif failure_probability > 0.6:
            recommendation = "Schedule maintenance within 24 hours"
        elif failure_probability > 0.4:
            recommendation = "Schedule maintenance within 1 week"
        elif failure_probability > 0.2:
            recommendation = "Monitor closely, schedule preventive maintenance"
        else:
            recommendation = "Equipment operating normally"
        
        # Save prediction to database in background
        background_tasks.add_task(
            save_prediction,
            sensor_input.machine_id,
            failure_probability,
            predicted_failure,
            ml_model.model_type,
            failure_probability
        )
        
        return PredictionResponse(
            machine_id=sensor_input.machine_id,
            failure_probability=failure_probability,
            predicted_failure=predicted_failure,
            recommendation=recommendation,
            confidence_score=failure_probability,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{machine_id}")
async def get_machine_predictions(machine_id: str, limit: int = 50):
    """Get prediction history for a specific machine"""
    try:
        predictions = get_predictions(machine_id, limit)
        return [
            {
                "id": pred.id,
                "machine_id": pred.machine_id,
                "prediction_date": pred.prediction_date,
                "failure_probability": pred.failure_probability,
                "predicted_failure": pred.predicted_failure,
                "model_used": pred.model_used,
                "confidence_score": pred.confidence_score,
                "recommended_action": pred.recommended_action
            }
            for pred in predictions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/maintenance/schedule")
async def schedule_maintenance(schedule_input: MaintenanceScheduleInput):
    """Create a new maintenance schedule"""
    try:
        schedule = create_maintenance_schedule(
            machine_id=schedule_input.machine_id,
            scheduled_date=schedule_input.scheduled_date,
            maintenance_type=schedule_input.maintenance_type,
            priority=schedule_input.priority,
            assigned_technician=schedule_input.assigned_technician,
            estimated_duration=schedule_input.estimated_duration,
            cost_estimate=schedule_input.cost_estimate,
            notes=schedule_input.notes
        )
        
        return {
            "id": schedule.id,
            "machine_id": schedule.machine_id,
            "scheduled_date": schedule.scheduled_date,
            "maintenance_type": schedule.maintenance_type,
            "priority": schedule.priority,
            "status": schedule.status,
            "message": "Maintenance scheduled successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/maintenance/schedule")
async def get_maintenance_schedules(status: Optional[str] = None, priority: Optional[str] = None):
    """Get maintenance schedule with optional filters"""
    try:
        schedules = get_maintenance_schedule(status, priority)
        return [
            {
                "id": sched.id,
                "machine_id": sched.machine_id,
                "scheduled_date": sched.scheduled_date,
                "maintenance_type": sched.maintenance_type,
                "priority": sched.priority,
                "status": sched.status,
                "assigned_technician": sched.assigned_technician,
                "estimated_duration": sched.estimated_duration,
                "cost_estimate": sched.cost_estimate,
                "notes": sched.notes,
                "created_at": sched.created_at
            }
            for sched in schedules
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_dashboard_data():
    """Get dashboard summary data"""
    try:
        # Get equipment summary
        equipment = get_equipment_list()
        equipment_count = len(equipment)
        
        # Get recent predictions
        recent_predictions = get_predictions(limit=100)
        high_risk_machines = [
            pred for pred in recent_predictions 
            if pred.failure_probability > 0.6
        ]
        
        # Get maintenance schedule
        upcoming_maintenance = get_maintenance_schedule(status="scheduled")
        overdue_maintenance = [
            maint for maint in upcoming_maintenance
            if maint.scheduled_date < datetime.utcnow()
        ]
        
        # Calculate equipment health scores
        equipment_health = {}
        for eq in equipment:
            machine_predictions = [
                pred for pred in recent_predictions 
                if pred.machine_id == eq.machine_id
            ]
            if machine_predictions:
                latest_prediction = machine_predictions[0]
                health_score = (1 - latest_prediction.failure_probability) * 100
                equipment_health[eq.machine_id] = {
                    "health_score": round(health_score, 2),
                    "status": eq.status,
                    "machine_type": eq.machine_type,
                    "failure_probability": latest_prediction.failure_probability
                }
        
        return {
            "summary": {
                "total_equipment": equipment_count,
                "high_risk_machines": len(high_risk_machines),
                "upcoming_maintenance": len(upcoming_maintenance),
                "overdue_maintenance": len(overdue_maintenance)
            },
            "equipment_health": equipment_health,
            "recent_alerts": [
                {
                    "machine_id": pred.machine_id,
                    "failure_probability": pred.failure_probability,
                    "recommended_action": pred.recommended_action,
                    "prediction_date": pred.prediction_date
                }
                for pred in high_risk_machines[:10]
            ],
            "maintenance_schedule": [
                {
                    "machine_id": maint.machine_id,
                    "scheduled_date": maint.scheduled_date,
                    "maintenance_type": maint.maintenance_type,
                    "priority": maint.priority,
                    "status": maint.status
                }
                for maint in upcoming_maintenance[:10]
            ]
        }
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)