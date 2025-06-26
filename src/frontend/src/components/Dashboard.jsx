import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

function Dashboard() {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Simulated historical data fetch
    const historicalData = [
      { time: '2025-06-01 00:00', probability: 0.1 },
      { time: '2025-06-01 01:00', probability: 0.15 },
      { time: '2025-06-01 02:00', probability: 0.2 },
      { time: '2025-06-01 03:00', probability: 0.8 },
    ];
    setData(historicalData);
  }, []);

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Failure Probability Trend</h2>
      <Plot
        data={[
          {
            x: data.map(d => d.time),
            y: data.map(d => d.probability * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Failure Probability (%)',
          },
        ]}
        layout={{
          xaxis: { title: 'Time' },
          yaxis: { title: 'Failure Probability (%)', range: [0, 100] },
          height: 400,
        }}
      />
    </div>
  );
}

export default Dashboard;