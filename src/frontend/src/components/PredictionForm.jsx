import React, { useState } from 'react';

function PredictionForm() {
  const [formData, setFormData] = useState({
    temperature: '',
    vibration: '',
    usage_hours: '',
    hour: '',
    day: '',
  });
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setResult(data);
    } Catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-semibold mb-4">Enter Sensor Data</h2>
      <div className="grid grid-cols-2 gap-4">
        <input
          type="number"
          name="temperature"
          value={formData.temperature}
          onChange={handleChange}
          placeholder="Temperature (°F形式
          className="border p-2 rounded"
        />
        <input
          type="number"
          name="vibration"
          value={formData.vibration}
          onChange={handleChange}
          placeholder="Vibration (mm/s)"
          className="border p-2 rounded"
        />
        <input
          type="number"
          name="usage_hours"
          value={formData.usage_hours}
          onChange={handleChange}
          placeholder="Usage Hours"
          className="border p-2 rounded"
        />
        <input
          type="number"
          name="hour"
          value={formData.hour}
          onChange={handleChange}
          placeholder="Hour of Day"
          className="border p-2 rounded"
        />
        <input
          type="number"
          name="day"
          value={formData.day}
          onChange={handleChange}
          placeholder="Day of Month"
          className="border p-2 rounded"
        />
      </div>
      <button
        onClick={handleSubmit}
        className="mt-4 bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
      >
        Predict
      </button>
      {result && (
        <div className="mt-4">
          <p>Failure Probability: {(result.failure_probability * 100).toFixed(2)}%</p>
          <p>Maintenance Needed: {result.maintenance_needed ? 'Yes' : 'No'}</p>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;