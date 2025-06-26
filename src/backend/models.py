from pydantic import BaseModel

class SensorData(BaseModel):
    temperature: float
    vibration: float
    usage_hours: float
    hour: int
    day: int

class PredictionResponse(BaseModel):
    failure_probability: float
    maintenance_needed: bool