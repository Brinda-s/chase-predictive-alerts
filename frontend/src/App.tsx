import React, { useEffect, useState } from "react";
import TransactionsTable from "./components/TransactionsTable";

interface Transaction {
  id: number;
  date: string;
  description: string;
  amount: number;
  balance: number;
}

function App() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/transactions/?limit=10")
      .then((res) => res.json())
      .then((data) => setTransactions(data))
      .catch((err) => console.error("Error fetching transactions:", err));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Predictive Alerts Dashboard</h1>
      <TransactionsTable transactions={transactions} />
    </div>
  );
}

export default App;
