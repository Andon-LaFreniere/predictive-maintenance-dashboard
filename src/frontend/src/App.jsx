import React from 'react';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-center mb-6">Predictive Maintenance Dashboard</h1>
      <div className="max-w-4xl mx-auto">
        <PredictionForm />
        <Dashboard />
      </div>
    </div>
  );
}

export default App;