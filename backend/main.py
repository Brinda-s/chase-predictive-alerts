from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sys
import os

# Add ml-models to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-models'))

from load_transactions import load_transactions, fetch_transactions

print("Imports are fine ✅")

# Initialize FastAPI app
app = FastAPI(title="Chase Predictive Alerts API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing endpoints
@app.get("/")
def root():
    return {"message": "Predictive Alerts API is running 🚀"}

@app.post("/upload-transactions/")
def upload_transactions(file: UploadFile = File(...)):
    file_location = "transactions_temp.csv"
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())
    load_transactions(file_location)
    return {"message": "Transactions loaded successfully!"}

@app.get("/transactions/")
def get_transactions(limit: int = 100):
    return fetch_transactions(limit)

# SIMPLIFIED ML ENDPOINTS

@app.get("/predict-balance/{customer_id}")
async def predict_balance(customer_id: str, days_ahead: int = 7):
    """
    Predict future balance using pattern analysis
    """
    try:
        # Get customer transactions
        all_transactions = fetch_transactions(10000)
        customer_transactions = [t for t in all_transactions if t['customer'] == customer_id]
        
        if not customer_transactions:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        if len(customer_transactions) < 5:
            raise HTTPException(status_code=400, detail="Insufficient transaction history")
        
        # Simple pattern analysis
        import pandas as pd
        df = pd.DataFrame(customer_transactions)
        df = df.sort_values('id')
        
        current_balance = float(df['balance'].iloc[-1])
        recent_amounts = df['amount'].tail(10).tolist()
        avg_daily_change = sum(recent_amounts) / len(recent_amounts)
        
        # Generate predictions
        predictions = []
        predicted_balance = current_balance
        
        for day in range(1, days_ahead + 1):
            predicted_balance += avg_daily_change
            predictions.append({
                "day": day,
                "predicted_balance": round(predicted_balance, 2),
                "date": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + 
                        pd.Timedelta(days=day)).isoformat()
            })
        
        return {
            "customer_id": customer_id,
            "current_balance": current_balance,
            "predictions": predictions,
            "model_type": "pattern_analysis",
            "avg_daily_change": round(avg_daily_change, 2),
            "confidence": "medium"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/generate-alerts/{customer_id}")
async def generate_alerts(customer_id: str):
    """
    Generate rule-based alerts for a customer
    """
    try:
        # Get customer transactions
        all_transactions = fetch_transactions(10000)
        customer_transactions = [t for t in all_transactions if t['customer'] == customer_id]
        
        if not customer_transactions:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        import pandas as pd
        df = pd.DataFrame(customer_transactions)
        df = df.sort_values('id')
        
        alerts = []
        current_balance = float(df['balance'].iloc[-1])
        
        # Balance-based alerts
        if current_balance < 0:
            alerts.append({
                "type": "negative_balance",
                "severity": "high",
                "message": f"Account balance is negative: ${current_balance:.2f}",
                "customer": customer_id,
                "date": datetime.now().isoformat()
            })
        elif current_balance < 1000:
            alerts.append({
                "type": "low_balance",
                "severity": "medium",
                "message": f"Low account balance: ${current_balance:.2f}",
                "customer": customer_id,
                "date": datetime.now().isoformat()
            })
        
        # Large transaction alerts
        if len(df) >= 5:
            avg_transaction = df['amount'].abs().mean()
            recent_transactions = df.tail(5)
            
            for _, txn in recent_transactions.iterrows():
                if abs(txn['amount']) > avg_transaction * 2.5:
                    severity = "high" if abs(txn['amount']) > avg_transaction * 4 else "medium"
                    alerts.append({
                        "type": "large_transaction",
                        "severity": severity,
                        "message": f"Large transaction: {txn['description']} for ${txn['amount']:.2f}",
                        "customer": customer_id,
                        "date": datetime.now().isoformat()
                    })
        
        # Spending trend alerts
        if len(df) >= 10:
            recent_spending = df[df['amount'] < 0]['amount'].tail(5).sum()
            if recent_spending < -5000:
                alerts.append({
                    "type": "high_spending",
                    "severity": "medium",
                    "message": f"High recent spending: ${abs(recent_spending):.2f} in last 5 transactions",
                    "customer": customer_id,
                    "date": datetime.now().isoformat()
                })
        
        return {
            "customer_id": customer_id,
            "total_alerts": len(alerts),
            "alerts": alerts,
            "generated_at": datetime.now().isoformat(),
            "alert_types": list(set([alert["type"] for alert in alerts]))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")

@app.get("/detect-anomalies/{customer_id}")
async def detect_anomalies(customer_id: str):
    """
    Simple statistical anomaly detection
    """
    try:
        all_transactions = fetch_transactions(10000)
        customer_transactions = [t for t in all_transactions if t['customer'] == customer_id]
        
        if not customer_transactions:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(customer_transactions)
        
        # Statistical anomaly detection using Z-score
        amounts = df['amount'].abs()
        mean_amount = amounts.mean()
        std_amount = amounts.std()
        
        if std_amount == 0:
            threshold = mean_amount * 2
        else:
            threshold = mean_amount + (2 * std_amount)
        
        anomalies = df[df['amount'].abs() > threshold].copy()
        anomalies['anomaly_score'] = (anomalies['amount'].abs() - mean_amount) / std_amount if std_amount > 0 else 1.0
        anomalies['is_anomaly'] = True
        
        return {
            "customer_id": customer_id,
            "total_transactions": len(df),
            "anomalies_found": len(anomalies),
            "anomaly_rate": round(len(anomalies) / len(df), 3) if len(df) > 0 else 0,
            "threshold_amount": round(threshold, 2),
            "mean_transaction": round(mean_amount, 2),
            "anomalies": [
                {
                    "id": int(row['id']),
                    "description": row['description'],
                    "amount": float(row['amount']),
                    "balance": float(row['balance']),
                    "anomaly_score": round(float(row['anomaly_score']), 2),
                    "date": row['date']
                }
                for _, row in anomalies.iterrows()
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

# Health check for system
@app.get("/ml-status")
async def ml_status():
    """
    Check system status
    """
    return {
        "system_ready": True,
        "prediction_method": "pattern_analysis",
        "alert_method": "rule_based",
        "anomaly_method": "statistical",
        "endpoints_available": [
            "/predict-balance/{customer_id}",
            "/generate-alerts/{customer_id}",
            "/detect-anomalies/{customer_id}"
        ]
    }

# Get all customers
@app.get("/customers")
async def get_customers():
    """
    Get list of all customers
    """
    try:
        all_transactions = fetch_transactions(10000)
        customers = list(set([t['customer'] for t in all_transactions]))
        
        # Get customer stats
        customer_stats = {}
        for customer in customers:
            customer_txns = [t for t in all_transactions if t['customer'] == customer]
            customer_stats[customer] = {
                "transaction_count": len(customer_txns),
                "current_balance": customer_txns[-1]['balance'] if customer_txns else 0,
                "customer_type": customer.split('_')[0] if '_' in customer else "unknown"
            }
        
        return {
            "customers": sorted(customers),
            "total_customers": len(customers),
            "customer_details": customer_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch customers: {str(e)}")

# Customer summary
@app.get("/customer-summary/{customer_id}")
async def customer_summary(customer_id: str):
    """
    Get comprehensive customer summary
    """
    try:
        all_transactions = fetch_transactions(10000)
        customer_transactions = [t for t in all_transactions if t['customer'] == customer_id]
        
        if not customer_transactions:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        import pandas as pd
        df = pd.DataFrame(customer_transactions)
        
        # Calculate statistics
        current_balance = float(df['balance'].iloc[-1])
        total_transactions = len(df)
        total_credits = len(df[df['amount'] > 0])
        total_debits = len(df[df['amount'] < 0])
        avg_transaction = df['amount'].mean()
        largest_credit = df[df['amount'] > 0]['amount'].max() if total_credits > 0 else 0
        largest_debit = df[df['amount'] < 0]['amount'].min() if total_debits > 0 else 0
        
        return {
            "customer_id": customer_id,
            "current_balance": current_balance,
            "total_transactions": total_transactions,
            "total_credits": total_credits,
            "total_debits": total_debits,
            "avg_transaction_amount": round(avg_transaction, 2),
            "largest_credit": float(largest_credit) if largest_credit else 0,
            "largest_debit": float(largest_debit) if largest_debit else 0,
            "transaction_types": df['description'].value_counts().to_dict(),
            "account_type": customer_id.split('_')[0] if '_' in customer_id else "unknown"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customer summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)