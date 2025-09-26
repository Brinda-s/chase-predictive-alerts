from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from load_transactions import load_transactions, fetch_transactions

print("Imports are fine ✅")
# 1️⃣ Create FastAPI instance
app = FastAPI()

# 2️⃣ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Root endpoint
@app.get("/")
def root():
    return {"message": "Predictive Alerts API is running 🚀"}

# 4️⃣ Upload transactions endpoint
@app.post("/upload-transactions/")
def upload_transactions(file: UploadFile = File(...)):
    file_location = "transactions_temp.csv"
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())
    load_transactions(file_location)
    return {"message": "Transactions loaded successfully!"}

# 5️⃣ Fetch transactions endpoint
@app.get("/transactions/")
def get_transactions(limit: int = 100):
    return fetch_transactions(limit)
