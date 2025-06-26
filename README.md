# Machine Learning-Powered Predictive Maintenance System

A comprehensive industrial predictive maintenance system that uses machine learning to predict equipment failures before they occur. This system helps organizations optimize maintenance schedules, reduce unplanned downtime, and improve operational efficiency.

## Overview

This project demonstrates a complete end-to-end solution for predictive maintenance in industrial settings. It combines IoT sensor data simulation, machine learning algorithms, and a modern web interface to provide real-time equipment monitoring and failure prediction capabilities.

## Key Features

**Machine Learning Models**
- Random Forest and XGBoost classifiers for failure prediction
- Advanced feature engineering including anomaly detection and efficiency metrics
- Hyperparameter tuning with cross-validation for optimal performance
- Real-time prediction API with confidence scoring

**Data Management**
- Simulated IoT sensor data generation for 50+ industrial machines
- SQLite database with equipment registry, sensor readings, and maintenance history
- Time-series data processing with rolling averages and trend analysis
- Automated data pipeline for continuous model training

**Web Interface**
- Interactive dashboard with real-time equipment health monitoring
- Plotly.js visualizations for trend analysis and risk assessment
- Alert system with prioritized maintenance recommendations
- REST API with automatic OpenAPI documentation

**Industrial Applications**
- Equipment health scoring and risk stratification
- Predictive maintenance scheduling based on failure probability
- Historical trend analysis and pattern recognition
- Integration-ready API for existing enterprise systems

## Technical Architecture

The system is built using modern web technologies and machine learning frameworks:

- **Backend**: FastAPI with SQLAlchemy for database operations
- **Machine Learning**: scikit-learn and XGBoost for predictive modeling
- **Frontend**: HTML/JavaScript with Plotly.js for data visualization
- **Database**: SQLite for development, easily configurable for PostgreSQL/MySQL
- **Deployment**: Docker containerization with docker-compose support

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize System**
   ```bash
   python setup.py
   ```
   This command will:
   - Generate 10,000 sample sensor data records
   - Train both Random Forest and XGBoost models
   - Initialize the database with sample equipment
   - Set up all necessary directories

3. **Start the Application**
   ```bash
   python main.py
   ```

4. **Access the System**
   - Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## System Components

### Data Generation
The `data_generator.py` module creates realistic sensor data for five types of industrial equipment:
- Pumps, Motors, Compressors, Generators, and Conveyors
- Simulates normal operation and failure scenarios
- Includes temperature, vibration, pressure, rotation speed, and current readings
- Generates time-based features and maintenance history

### Machine Learning Pipeline
The `ml_model.py` module implements the predictive analytics engine:
- Feature engineering with interaction terms and anomaly detection
- Model comparison between Random Forest and XGBoost algorithms
- Grid search hyperparameter optimization
- Model persistence and loading for production use

### Database Layer
The `database.py` module manages all data operations:
- Equipment registry with installation dates and specifications
- Time-series sensor data storage
- Prediction history and confidence tracking
- Maintenance scheduling and completion records

### API Backend
The `main.py` module provides the REST API interface:
- Real-time prediction endpoints for failure analysis
- Equipment management and sensor data retrieval
- Dashboard data aggregation and summary statistics
- Maintenance scheduling and workflow management

### Web Dashboard
The frontend provides an intuitive interface for:
- Equipment health monitoring with color-coded risk levels
- Interactive charts showing trends and distributions
- Alert management with prioritized action items
- Manual prediction input for what-if analysis

## API Endpoints

### Core Operations
- `GET /` - Interactive dashboard interface
- `GET /health` - System health and status check
- `POST /predict` - Run failure prediction on sensor data
- `GET /dashboard` - Comprehensive system overview

### Equipment Management
- `GET /equipment` - List all registered equipment
- `GET /equipment/{machine_id}/sensor-data` - Historical sensor readings
- `GET /predictions/{machine_id}` - Prediction history for specific equipment

### Maintenance Operations
- `POST /maintenance/schedule` - Create new maintenance tasks
- `GET /maintenance/schedule` - View scheduled maintenance activities

## Machine Learning Details

### Model Performance
Both models are trained using stratified cross-validation with the following characteristics:
- **Random Forest**: Provides feature importance analysis and robust performance
- **XGBoost**: Optimized for imbalanced datasets with superior predictive accuracy
- **Evaluation Metrics**: ROC-AUC, precision, recall, and F1-score
- **Class Imbalance Handling**: Weighted classes and specialized sampling techniques

### Feature Engineering
The system creates derived features to improve prediction accuracy:
- **Anomaly Detection**: Flags for temperature and vibration spikes
- **Efficiency Metrics**: Equipment performance ratios and operational scores
- **Time Features**: Hour, day, and seasonal patterns
- **Rolling Statistics**: Moving averages for trend detection
- **Interaction Terms**: Cross-feature relationships and correlations

## Production Deployment

### Docker Deployment
```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

### Docker Compose
```bash
docker-compose up -d
```

### Production Considerations
For production environments, consider these enhancements:
- Replace SQLite with PostgreSQL or MySQL for scalability
- Implement proper authentication and authorization
- Add logging, monitoring, and alerting systems
- Set up automated model retraining pipelines
- Configure load balancing and high availability
- Implement data encryption and security measures

## Use Cases and Benefits

### Manufacturing
- Reduce unplanned downtime by predicting motor and pump failures
- Optimize maintenance schedules based on actual equipment condition
- Lower maintenance costs through condition-based interventions

### Transportation
- Predict vehicle component failures before they cause breakdowns
- Schedule maintenance during planned downtime periods
- Improve fleet availability and operational efficiency

### Energy and Utilities
- Monitor critical infrastructure for early failure detection
- Prevent cascading failures in power generation systems
- Optimize maintenance resources and workforce allocation

### Process Industries
- Predict failures in pumps, compressors, and rotating equipment
- Minimize safety risks through proactive maintenance
- Reduce spare parts inventory through better demand forecasting

## Customization and Extension

### Adding New Equipment Types
Modify the `data_generator.py` file to include additional machine types with their specific operating characteristics and failure patterns.

### Custom Feature Engineering
Extend the feature engineering pipeline in `ml_model.py` to incorporate domain-specific knowledge and additional sensor types.

### Advanced Models
The modular architecture supports integration of advanced algorithms such as:
- LSTM networks for time-series prediction
- Isolation Forest for anomaly detection
- Ensemble methods combining multiple algorithms

### Integration Options
The REST API design facilitates integration with:
- Enterprise Resource Planning (ERP) systems
- Computerized Maintenance Management Systems (CMMS)
- Industrial IoT platforms and data historians
- Business intelligence and reporting tools

## Performance and Scalability

The system is designed to handle:
- Thousands of equipment units with real-time monitoring
- High-frequency sensor data ingestion and processing
- Concurrent prediction requests from multiple users
- Historical data analysis and trend reporting

For large-scale deployments, the architecture supports:
- Horizontal scaling with load balancers
- Database clustering and replication
- Microservices decomposition
- Cloud-native deployment patterns

## Support and Maintenance

### System Requirements
- Memory: 4GB RAM minimum, 8GB recommended
- Storage: 10GB available space for models and data
- Network: Standard HTTP/HTTPS connectivity
- Operating System: Windows, macOS, or Linux

### Troubleshooting
Common issues and solutions:
- Model training failures: Check data quality and feature distributions
- Database connectivity: Verify file permissions and disk space
- API performance: Monitor memory usage and optimize queries
- Dashboard loading: Check browser compatibility and network connectivity

## License and Contributing

This project is available under the MIT License, allowing for both academic and commercial use. Contributions are welcome through standard GitHub workflows including issue reporting, feature requests, and pull requests.

For technical support or questions about implementation, please refer to the API documentation at `/docs` when the system is running, or review the inline code comments and docstrings throughout the codebase.
