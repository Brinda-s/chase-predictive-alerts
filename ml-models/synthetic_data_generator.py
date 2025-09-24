import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_transactions(n_customers=1000, n_transactions=100000):
    # Use customer IDs instead of unique names
    customers = [f"cust_{i+1}_{fake.first_name()}" for i in range(n_customers)]
    records = []
    
    for cust in customers:
        balance = random.randint(500, 5000)  # initial balance
        start_date = datetime(2023, 1, 1)
        
        for _ in range(n_transactions // n_customers):
            days_offset = random.randint(0, 365)
            txn_date = start_date + timedelta(days=days_offset)

            txn_type = random.choice(["salary", "bill", "purchase", "seasonal"])
            
            if txn_type == "salary":
                amount = random.randint(2000, 5000)
                balance += amount
                description = "Salary Credit"
            
            elif txn_type == "bill":
                amount = random.randint(50, 500)
                balance -= amount
                description = f"{random.choice(['Electricity', 'Water', 'Internet', 'Rent'])} Bill"
            
            elif txn_type == "purchase":
                amount = random.randint(10, 200)
                balance -= amount
                description = f"Purchase at {fake.company()}"
            
            else:  # seasonal spikes
                amount = random.randint(100, 1000)
                balance -= amount
                description = f"Seasonal Spending - {fake.word().capitalize()}"

            records.append({
                "customer": cust,
                "date": txn_date,
                "transaction_type": txn_type,
                "description": description,
                "amount": amount,
                "balance": balance
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(by=["customer", "date"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_transactions()
    print(df.head(20))
    df.to_csv("synthetic_transactions.csv", index=False)
    print("Synthetic dataset saved to synthetic_transactions.csv")
