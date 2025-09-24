import pandas as pd
from database import SessionLocal, Transaction

CSV_FILE = "transactions.csv"
CHUNK_SIZE = 1000  # rows per chunk

def load_transactions(file_path=CSV_FILE):
    df = pd.read_csv(file_path)

    df['amount'] = df['amount'].astype(float)
    df['balance'] = df['balance'].astype(float)
    df['date'] = pd.to_datetime(df['date']).dt.date

    session = SessionLocal()

    try:
        for start in range(0, len(df), CHUNK_SIZE):
            end = start + CHUNK_SIZE
            chunk = df.iloc[start:end]

            transactions_to_add = [
                Transaction(
                    customer=row['customer'],
                    date=row['date'],
                    description=row['description'],
                    amount=row['amount'],
                    balance=row['balance']
                ) for _, row in chunk.iterrows()
            ]

            session.bulk_save_objects(transactions_to_add)
            session.commit()
            print(f"Inserted {len(transactions_to_add)} transactions in this chunk.")

        print("All chunks processed.")

    except Exception as e:
        session.rollback()
        print("Error inserting transactions:", e)
    finally:
        session.close()


def fetch_transactions(limit: int = 100):
    session = SessionLocal()
    try:
        results = session.query(Transaction).limit(limit).all()
        # convert ORM objects to dict
        rows = [
            {
                "id": t.id,
                "customer": t.customer,
                "date": t.date.strftime("%Y-%m-%d"),
                "description": t.description,
                "amount": float(t.amount),
                "balance": float(t.balance)
            }
            for t in results
        ]
        return rows
    finally:
        session.close()
