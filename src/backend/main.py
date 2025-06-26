from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import SensorData, PredictionResponse
from utils import load_model_and_scaler, predict_failure

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model, scaler, features = load_model_and_scaler(
    'models/xgboost_model.pkl', 'models/scaler.pkl', 'models/features.pkl'
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    prob, maintenance_needed = predict_failure(model, scaler, features, data)
    return {"failure_probability": prob, "maintenance_needed": maintenance_needed}