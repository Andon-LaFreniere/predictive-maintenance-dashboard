import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sensor_data(num_samples=10000, num_machines=50):
    """
    Generate simulated IoT sensor data for predictive maintenance
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    machine_ids = [f"MACHINE_{i:03d}" for i in range(1, num_machines + 1)]
    
    # Define machine types with different failure patterns
    machine_types = ["Pump", "Motor", "Compressor", "Generator", "Conveyor"]
    
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_samples):
        machine_id = random.choice(machine_ids)
        machine_type = random.choice(machine_types)
        
        # Simulate time series data
        timestamp = start_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Base values for different machine types
        base_temp = {"Pump": 45, "Motor": 55, "Compressor": 65, "Generator": 70, "Conveyor": 40}
        base_vibration = {"Pump": 2.5, "Motor": 3.0, "Compressor": 4.0, "Generator": 2.0, "Conveyor": 1.5}
        base_pressure = {"Pump": 15, "Motor": 8, "Compressor": 25, "Generator": 5, "Conveyor": 10}
        
        # Simulate normal operation vs failure conditions
        failure_probability = random.random()
        is_failure_approaching = failure_probability > 0.85  # 15% failure rate
        
        # Generate sensor readings with noise and failure patterns
        if is_failure_approaching:
            # Elevated readings indicating potential failure
            temperature = base_temp[machine_type] + np.random.normal(15, 5)
            vibration = base_vibration[machine_type] + np.random.normal(2, 0.5)
            pressure = base_pressure[machine_type] + np.random.normal(-3, 1)
            rotation_speed = 1800 + np.random.normal(-200, 50)
            current = 25 + np.random.normal(5, 2)
            failure = 1
        else:
            # Normal operation
            temperature = base_temp[machine_type] + np.random.normal(0, 3)
            vibration = base_vibration[machine_type] + np.random.normal(0, 0.3)
            pressure = base_pressure[machine_type] + np.random.normal(0, 1)
            rotation_speed = 1800 + np.random.normal(0, 30)
            current = 25 + np.random.normal(0, 1)
            failure = 0
        
        # Calculate derived features
        operating_hours = random.randint(1000, 50000)
        maintenance_count = operating_hours // 2000 + random.randint(0, 3)
        
        data.append({
            'timestamp': timestamp,
            'machine_id': machine_id,
            'machine_type': machine_type,
            'temperature': max(0, temperature),
            'vibration': max(0, vibration),
            'pressure': max(0, pressure),
            'rotation_speed': max(0, rotation_speed),
            'current': max(0, current),
            'operating_hours': operating_hours,
            'maintenance_count': maintenance_count,
            'failure': failure
        })
    
    df = pd.DataFrame(data)
    
    # Add some time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Calculate rolling averages for trend analysis
    df = df.sort_values(['machine_id', 'timestamp'])
    df['temp_rolling_avg'] = df.groupby('machine_id')['temperature'].rolling(window=5, min_periods=1).mean().values
    df['vibration_rolling_avg'] = df.groupby('machine_id')['vibration'].rolling(window=5, min_periods=1).mean().values
    
    return df

def save_dataset(df, filename='equipment_data.csv'):
    """Save the generated dataset to CSV"""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    print(f"Dataset shape: {df.shape}")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    return df

if __name__ == "__main__":
    # Generate and save dataset
    df = generate_sensor_data()
    save_dataset(df)
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nFailure distribution by machine type:")
    print(df.groupby('machine_type')['failure'].agg(['count', 'sum', 'mean']))