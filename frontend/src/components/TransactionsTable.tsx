import React from "react";

interface Transaction {
  id: number;
  date: string;
  description: string;
  amount: number;
  balance: number;
}

interface Props {
  transactions: Transaction[];
}

const TransactionsTable: React.FC<Props> = ({ transactions }) => {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Transactions</h2>
      <table className="min-w-full border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
            <th className="border px-4 py-2">Date</th>
            <th className="border px-4 py-2">Description</th>
            <th className="border px-4 py-2">Amount ($)</th>
            <th className="border px-4 py-2">Balance ($)</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((txn) => (
            <tr key={txn.id}>
              <td className="border px-4 py-2">{txn.date}</td>
              <td className="border px-4 py-2">{txn.description}</td>
              <td className="border px-4 py-2">{txn.amount}</td>
              <td className="border px-4 py-2">{txn.balance}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TransactionsTable;
