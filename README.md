# Predictive Balance Alerts
Features

Balance Forecasting: Random Forest model predicts 7-day balance with R² = 0.997
Anomaly Detection: Isolation Forest identifies unusual spending patterns
Real-time API: FastAPI backend with transaction processing
React Dashboard: TypeScript frontend with Tailwind CSS

Architecture
backend/     # FastAPI + SQLite + ML inference
frontend/    # React + TypeScript + Tailwind
ml-models/   # scikit-learn models (balance prediction, anomaly detection)  
data/        # 210K synthetic transactions, 1000 customers
Quick Start
bash# Backend
cd backend && python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn scikit-learn pandas numpy joblib
uvicorn main:app --reload

# Frontend  
cd frontend && npm install && npm run dev

# Train Models
cd ml-models && python predictive_models.py
Performance

Model Accuracy: 99.7% (MAE: $1,444)
Training Data: 210K transactions, 3+ years
Prediction Latency: <100ms
Anomaly Detection: 10.1% flagging rate

API Endpoints
GET  /                           # Health check
POST /upload-transactions/       # Bulk upload  
GET  /transactions/?limit=N      # Retrieve data
Demo Output
Generated 5 alerts for customer:
  Unusual Water Bill: $13,055 (anomaly detected)
  Large Seasonal Expense: $10,809  
  High Electricity Bill: $14,475
  Irregular Purchase: -$376
  Credit Refund: -$2,068
Stack
FastAPI • React • TypeScript • scikit-learn • SQLite • Tailwind CSS
Author
@Brinda-s