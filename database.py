from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./maintenance.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Equipment(Base):
    __tablename__ = "equipment"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, unique=True, index=True)
    machine_type = Column(String)
    location = Column(String)
    installation_date = Column(DateTime)
    last_maintenance = Column(DateTime)
    status = Column(String, default="operational")
    created_at = Column(DateTime, default=datetime.utcnow)

class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    temperature = Column(Float)
    vibration = Column(Float)
    pressure = Column(Float)
    rotation_speed = Column(Float)
    current = Column(Float)
    operating_hours = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class MaintenancePrediction(Base):
    __tablename__ = "maintenance_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    prediction_date = Column(DateTime, index=True)
    failure_probability = Column(Float)
    predicted_failure = Column(Boolean)
    model_used = Column(String)
    confidence_score = Column(Float)
    recommended_action = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class MaintenanceSchedule(Base):
    __tablename__ = "maintenance_schedule"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    scheduled_date = Column(DateTime)
    maintenance_type = Column(String)  # 'preventive', 'predictive', 'corrective'
    priority = Column(String)  # 'low', 'medium', 'high', 'critical'
    status = Column(String, default="scheduled")  # 'scheduled', 'in_progress', 'completed', 'cancelled'
    assigned_technician = Column(String)
    estimated_duration = Column(Float)  # in hours
    cost_estimate = Column(Float)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database utility functions
def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

def populate_sample_equipment():
    """Populate sample equipment data"""
    db = SessionLocal()
    
    # Check if equipment already exists
    existing_equipment = db.query(Equipment).first()
    if existing_equipment:
        print("Sample equipment already exists")
        db.close()
        return
    
    sample_equipment = [
        {
            "machine_id": "MACHINE_001",
            "machine_type": "Pump",
            "location": "Building A - Floor 1",
            "installation_date": datetime(2020, 1, 15),
            "last_maintenance": datetime(2024, 11, 1)
        },
        {
            "machine_id": "MACHINE_002",
            "machine_type": "Motor",
            "location": "Building A - Floor 2",
            "installation_date": datetime(2019, 8, 20),
            "last_maintenance": datetime(2024, 10, 15)
        },
        {
            "machine_id": "MACHINE_003",
            "machine_type": "Compressor",
            "location": "Building B - Floor 1",
            "installation_date": datetime(2021, 3, 10),
            "last_maintenance": datetime(2024, 12, 1)
        },
        {
            "machine_id": "MACHINE_004",
            "machine_type": "Generator",
            "location": "Building B - Floor 2",
            "installation_date": datetime(2018, 11, 5),
            "last_maintenance": datetime(2024, 9, 20)
        },
        {
            "machine_id": "MACHINE_005",
            "machine_type": "Conveyor",
            "location": "Warehouse",
            "installation_date": datetime(2022, 2, 1),
            "last_maintenance": datetime(2024, 11, 10)
        }
    ]
    
    for equipment_data in sample_equipment:
        equipment = Equipment(**equipment_data)
        db.add(equipment)
    
    db.commit()
    db.close()
    print("Sample equipment data populated successfully")

def load_sensor_data_to_db(csv_file='equipment_data.csv'):
    """Load sensor data from CSV to database"""
    db = SessionLocal()
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check if data already exists
        existing_data = db.query(SensorData).first()
        if existing_data:
            print("Sensor data already exists in database")
            db.close()
            return
        
        # Convert DataFrame to database records
        records = []
        for _, row in df.iterrows():
            sensor_record = SensorData(
                machine_id=row['machine_id'],
                timestamp=pd.to_datetime(row['timestamp']),
                temperature=row['temperature'],
                vibration=row['vibration'],
                pressure=row['pressure'],
                rotation_speed=row['rotation_speed'],
                current=row['current'],
                operating_hours=row['operating_hours']
            )
            records.append(sensor_record)
        
        # Bulk insert
        db.bulk_save_objects(records)
        db.commit()
        print(f"Loaded {len(records)} sensor data records to database")
        
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        db.rollback()
    finally:
        db.close()

def get_equipment_list():
    """Get list of all equipment"""
    db = SessionLocal()
    equipment = db.query(Equipment).all()
    db.close()
    return equipment

def get_sensor_data(machine_id=None, limit=1000):
    """Get sensor data for a specific machine or all machines"""
    db = SessionLocal()
    query = db.query(SensorData)
    
    if machine_id:
        query = query.filter(SensorData.machine_id == machine_id)
    
    sensor_data = query.order_by(SensorData.timestamp.desc()).limit(limit).all()
    db.close()
    return sensor_data

def save_prediction(machine_id, failure_probability, predicted_failure, model_used, confidence_score=None):
    """Save prediction to database"""
    db = SessionLocal()
    
    # Determine recommended action based on failure probability
    if failure_probability > 0.8:
        recommended_action = "Immediate maintenance required"
    elif failure_probability > 0.6:
        recommended_action = "Schedule maintenance within 24 hours"
    elif failure_probability > 0.4:
        recommended_action = "Schedule maintenance within 1 week"
    else:
        recommended_action = "Continue monitoring"
    
    prediction = MaintenancePrediction(
        machine_id=machine_id,
        prediction_date=datetime.utcnow(),
        failure_probability=failure_probability,
        predicted_failure=predicted_failure,
        model_used=model_used,
        confidence_score=confidence_score,
        recommended_action=recommended_action
    )
    
    db.add(prediction)
    db.commit()
    db.close()
    
    return prediction

def get_predictions(machine_id=None, limit=100):
    """Get predictions for a specific machine or all machines"""
    db = SessionLocal()
    query = db.query(MaintenancePrediction)
    
    if machine_id:
        query = query.filter(MaintenancePrediction.machine_id == machine_id)
    
    predictions = query.order_by(MaintenancePrediction.prediction_date.desc()).limit(limit).all()
    db.close()
    return predictions

def create_maintenance_schedule(machine_id, scheduled_date, maintenance_type, priority="medium", 
                              assigned_technician=None, estimated_duration=4.0, cost_estimate=500.0, notes=""):
    """Create a maintenance schedule entry"""
    db = SessionLocal()
    
    schedule = MaintenanceSchedule(
        machine_id=machine_id,
        scheduled_date=scheduled_date,
        maintenance_type=maintenance_type,
        priority=priority,
        assigned_technician=assigned_technician,
        estimated_duration=estimated_duration,
        cost_estimate=cost_estimate,
        notes=notes
    )
    
    db.add(schedule)
    db.commit()
    db.close()
    
    return schedule

def get_maintenance_schedule(status=None, priority=None):
    """Get maintenance schedule with optional filters"""
    db = SessionLocal()
    query = db.query(MaintenanceSchedule)
    
    if status:
        query = query.filter(MaintenanceSchedule.status == status)
    if priority:
        query = query.filter(MaintenanceSchedule.priority == priority)
    
    schedule = query.order_by(MaintenanceSchedule.scheduled_date).all()
    db.close()
    return schedule

if __name__ == "__main__":
    # Initialize database
    init_database()
    populate_sample_equipment()
    
    # Load sensor data if CSV exists
    try:
        load_sensor_data_to_db()
    except FileNotFoundError:
        print("equipment_data.csv not found. Run data_generator.py first.")
    
    print("Database setup completed!")