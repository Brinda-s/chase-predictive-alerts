import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)

os.makedirs("../data", exist_ok=True)

# Define personas
personas = [
    {"prefix":"student","count":500,"income":(500,1500),"expense":(300,800),"balance":(200,800)},
    {"prefix":"professional","count":1000,"income":(3000,7000),"expense":(1500,3000),"balance":(1000,5000)},
    {"prefix":"retired","count":500,"income":(1000,3000),"expense":(800,2000),"balance":(2000,8000)},
    {"prefix":"highnetworth","count":250,"income":(10000,20000),"expense":(4000,10000),"balance":(5000,20000)},
]

# Multi-year dates
start_date = datetime(2022,1,1)
end_date = datetime(2025,12,31)
dates = pd.date_range(start=start_date,end=end_date,freq="D")
n_days = len(dates)

records = []

for persona in personas:
    for i in range(persona["count"]):
        customer_id = f"{persona['prefix']}_{i+1}_{fake.first_name()}"
        balance = random.randint(*persona["balance"])

        # Generate random daily transactions for each customer
        for date in dates:
            txn_type = None
            description = ""
            amount = 0

            # Income on first day of month
            if date.day == 1:
                income = random.randint(*persona["income"])
                amount += income
                balance += income
                txn_type = "income"
                description = "Salary/Pension Credit"

            # Bills on 1,3,5
            if date.day in [1,3,5]:
                bill = random.randint(int(persona["expense"][0]/2), int(persona["expense"][1]/2))
                amount -= bill
                balance -= bill
                txn_type = "bill"
                description = f"{random.choice(['Rent','Electricity','Water','Internet'])} Bill"

            # Random purchases (~5%)
            if np.random.rand()<0.05:
                purchase = random.randint(10,500)
                amount -= purchase
                balance -= purchase
                txn_type = "purchase"
                description = f"Purchase at {fake.company()}"

            # Seasonal spikes (~10% in Dec, Jun)
            if date.month in [12,6] and np.random.rand()<0.1:
                seasonal = random.randint(100,2000)
                amount -= seasonal
                balance -= seasonal
                txn_type = "seasonal"
                description = f"Seasonal Expense - {fake.word().capitalize()}"

            if amount != 0:
                low_balance_flag = 1 if balance<500 else 0
                records.append({
                    "customer":customer_id,
                    "date":date,
                    "transaction_type":txn_type,
                    "description":description,
                    "amount":round(amount,2),
                    "balance":round(balance,2),
                    "low_balance_flag":low_balance_flag
                })

# Create DataFrame
df = pd.DataFrame(records)
df.sort_values(by=["customer","date"], inplace=True)
df.reset_index(drop=True,inplace=True)

# Save CSV
df.to_csv("../data/synthetic_transactions_large.csv",index=False)
print("✅ Large synthetic dataset generated: ../data/synthetic_transactions_large.csv")
print("Total transactions:", len(df))
