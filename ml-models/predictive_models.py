"""
CHASE PREDICTIVE BALANCE ALERTS - MACHINE LEARNING MODELS
Multi-layered prediction system with ensemble methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictiveBalanceSystem:
    """
    Main class for Chase Predictive Balance Alerts
    Handles balance forecasting, spending analysis, and alert generation
    """
    
    def __init__(self):
        self.balance_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.customer_patterns = {}
        
    def load_and_preprocess_data(self, file_path):
        """
        Load transaction data and create features for ML models
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by customer and date
        df = df.sort_values(['customer', 'date'])
        
        # Create time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create spending features
        df['amount_abs'] = df['amount'].abs()
        df['is_debit'] = (df['amount'] < 0).astype(int)
        df['is_credit'] = (df['amount'] > 0).astype(int)
        
        # Encode categorical features
        df['description_encoded'] = self.label_encoder.fit_transform(df['description'])
        
        print(f"Loaded {len(df)} transactions for {df['customer'].nunique()} customers")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def create_customer_features(self, df, customer_id):
        """
        Create customer-specific features for balance prediction
        """
        customer_data = df[df['customer'] == customer_id].copy()
        customer_data = customer_data.sort_values('date')
        
        # Create lagged features
        customer_data['balance_lag_1'] = customer_data['balance'].shift(1)
        customer_data['balance_lag_7'] = customer_data['balance'].shift(7)
        customer_data['amount_lag_1'] = customer_data['amount'].shift(1)
        
        # Rolling statistics
        customer_data['balance_ma_7'] = customer_data['balance'].rolling(window=7, min_periods=1).mean()
        customer_data['amount_ma_7'] = customer_data['amount'].rolling(window=7, min_periods=1).mean()
        customer_data['amount_std_7'] = customer_data['amount'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Days since last transaction
        customer_data['days_since_last'] = customer_data['date'].diff().dt.days.fillna(0)
        
        # Spending velocity (change in spending pattern)
        customer_data['spending_velocity'] = customer_data['amount'].diff().fillna(0)
        
        return customer_data.dropna()
    
    def train_balance_prediction_model(self, df):
        """
        Train Random Forest model to predict future balance
        """
        print("Training balance prediction model...")
        
        # Prepare training data for all customers
        all_features = []
        all_targets = []
        
        for customer in df['customer'].unique()[:5]:  # Train on first 5 customers for speed
            customer_data = self.create_customer_features(df, customer)
            
            if len(customer_data) < 10:  # Skip customers with too little data
                continue
                
            # Features for prediction
            feature_cols = [
                'balance_lag_1', 'balance_lag_7', 'amount_lag_1',
                'balance_ma_7', 'amount_ma_7', 'amount_std_7',
                'days_since_last', 'spending_velocity',
                'day_of_week', 'day_of_month', 'month',
                'is_weekend', 'amount_abs', 'is_debit', 'is_credit',
                'description_encoded'
            ]
            
            X = customer_data[feature_cols]
            y = customer_data['balance']
            
            all_features.append(X)
            all_targets.append(y)
        
        # Combine all customer data
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = pd.concat(all_targets, ignore_index=True)
        
        # Handle any remaining NaN values
        X_combined = X_combined.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_combined, test_size=0.2, random_state=42
        )
        
        # Train model
        self.balance_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.balance_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.balance_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Balance Prediction Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        return mae, r2
    
    def train_anomaly_detection(self, df):
        """
        Train anomaly detection model for unusual spending patterns
        """
        print("Training anomaly detection model...")
        
        # Features for anomaly detection
        anomaly_features = []
        
        for customer in df['customer'].unique()[:5]:
            customer_data = self.create_customer_features(df, customer)
            
            if len(customer_data) < 10:
                continue
                
            features = customer_data[[
                'amount_abs', 'balance', 'amount_ma_7', 'amount_std_7',
                'day_of_week', 'is_weekend'
            ]].fillna(0)
            
            anomaly_features.append(features)
        
        # Combine features
        X_anomaly = pd.concat(anomaly_features, ignore_index=True)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% of transactions to be anomalies
            random_state=42
        )
        
        self.anomaly_detector.fit(X_anomaly)
        
        # Test anomaly detection
        anomaly_scores = self.anomaly_detector.decision_function(X_anomaly)
        anomalies = self.anomaly_detector.predict(X_anomaly)
        
        print(f"Anomaly Detection Model trained")
        print(f"Detected {(anomalies == -1).sum()} anomalies out of {len(anomalies)} transactions")
        
    def predict_future_balance(self, customer_data, days_ahead=7):
        """
        Predict balance for a specific customer N days ahead
        """
        if self.balance_model is None:
            raise ValueError("Balance model not trained yet!")
        
        # Get latest customer data
        latest_data = customer_data.tail(1).copy()
        
        predictions = []
        current_balance = latest_data['balance'].iloc[0]
        
        for day in range(1, days_ahead + 1):
            # Create features for prediction
            feature_cols = [
                'balance_lag_1', 'balance_lag_7', 'amount_lag_1',
                'balance_ma_7', 'amount_ma_7', 'amount_std_7',
                'days_since_last', 'spending_velocity',
                'day_of_week', 'day_of_month', 'month',
                'is_weekend', 'amount_abs', 'is_debit', 'is_credit',
                'description_encoded'
            ]
            
            # Use latest available features
            X_pred = latest_data[feature_cols].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predict next balance
            predicted_balance = self.balance_model.predict(X_pred_scaled)[0]
            predictions.append({
                'day': day,
                'predicted_balance': predicted_balance,
                'date': latest_data['date'].iloc[0] + timedelta(days=day)
            })
            
            # Update for next prediction (simple approach)
            current_balance = predicted_balance
        
        return predictions
    
    def detect_spending_anomalies(self, customer_data):
        """
        Detect unusual spending patterns for alerts
        """
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained yet!")
        
        # Prepare features
        features = customer_data[[
            'amount_abs', 'balance', 'amount_ma_7', 'amount_std_7',
            'day_of_week', 'is_weekend'
        ]].fillna(0)
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.decision_function(features)
        is_anomaly = self.anomaly_detector.predict(features)
        
        # Add results to data
        customer_data = customer_data.copy()
        customer_data['anomaly_score'] = anomaly_scores
        customer_data['is_anomaly'] = (is_anomaly == -1)
        
        return customer_data
    
    def generate_predictive_alerts(self, customer_id, customer_data, days_ahead=7):
        """
        Generate intelligent alerts based on predictions
        """
        alerts = []
        
        # 1. Balance prediction alerts
        try:
            balance_predictions = self.predict_future_balance(customer_data, days_ahead)
            
            for pred in balance_predictions:
                if pred['predicted_balance'] < 0:
                    alerts.append({
                        'type': 'low_balance_prediction',
                        'severity': 'high',
                        'message': f"⚠️ Predicted negative balance of ${pred['predicted_balance']:.2f} on {pred['date'].strftime('%Y-%m-%d')}",
                        'date': pred['date'],
                        'customer': customer_id
                    })
                elif pred['predicted_balance'] < 1000:
                    alerts.append({
                        'type': 'low_balance_warning',
                        'severity': 'medium',
                        'message': f"📉 Low balance predicted: ${pred['predicted_balance']:.2f} on {pred['date'].strftime('%Y-%m-%d')}",
                        'date': pred['date'],
                        'customer': customer_id
                    })
        except Exception as e:
            print(f"Balance prediction failed: {e}")
        
        # 2. Anomaly detection alerts
        try:
            anomaly_data = self.detect_spending_anomalies(customer_data)
            recent_anomalies = anomaly_data[anomaly_data['is_anomaly'] == True].tail(5)
            
            for _, transaction in recent_anomalies.iterrows():
                alerts.append({
                    'type': 'unusual_spending',
                    'severity': 'medium',
                    'message': f"🔍 Unusual transaction detected: {transaction['description']} for ${transaction['amount']:.2f}",
                    'date': transaction['date'],
                    'customer': customer_id
                })
        except Exception as e:
            print(f"Anomaly detection failed: {e}")
        
        # 3. Pattern-based alerts
        current_balance = customer_data['balance'].iloc[-1]
        avg_monthly_spending = customer_data[customer_data['amount'] < 0]['amount'].mean() * 30
        
        if current_balance < abs(avg_monthly_spending):
            alerts.append({
                'type': 'cash_flow_warning',
                'severity': 'high',
                'message': f"💰 Current balance (${current_balance:.2f}) may not cover typical monthly expenses",
                'date': datetime.now(),
                'customer': customer_id
            })
        
        return alerts
    
    def save_models(self, model_dir="ml-models"):
        """
        Save trained models to disk
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.balance_model:
            joblib.dump(self.balance_model, f"{model_dir}/balance_model.pkl")
            joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
            print(f"Balance model saved to {model_dir}/balance_model.pkl")
        
        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, f"{model_dir}/anomaly_detector.pkl")
            print(f"Anomaly detector saved to {model_dir}/anomaly_detector.pkl")
    
    def load_models(self, model_dir="ml-models"):
        """
        Load trained models from disk
        """
        try:
            self.balance_model = joblib.load(f"{model_dir}/balance_model.pkl")
            self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
            self.anomaly_detector = joblib.load(f"{model_dir}/anomaly_detector.pkl")
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")


# DEMO USAGE
if __name__ == "__main__":
    # Initialize the prediction system
    predictor = PredictiveBalanceSystem()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data("../data/synthetic_transactions.csv")
    
    # Train models
    mae, r2 = predictor.train_balance_prediction_model(df)
    predictor.train_anomaly_detection(df)
    
    # Save models
    predictor.save_models()
    
    # Generate alerts for a specific customer
    customer_id = df['customer'].iloc[0]
    customer_data = predictor.create_customer_features(df, customer_id)
    
    alerts = predictor.generate_predictive_alerts(customer_id, customer_data)
    
    print(f"\n🚨 Generated {len(alerts)} alerts for {customer_id}:")
    for alert in alerts:
        print(f"  {alert['severity'].upper()}: {alert['message']}")
    
    print("\n✅ ML Models ready for integration!")