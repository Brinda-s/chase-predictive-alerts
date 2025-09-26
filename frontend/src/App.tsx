import React, { useState, useEffect } from 'react';
import './App.css';

interface Transaction {
  id: number;
  customer: string;
  date: string;
  description: string;
  amount: number;
  balance: number;
}

interface Prediction {
  day: number;
  predicted_balance: number;
  date: string;
}

interface Alert {
  type: string;
  severity: string;
  message: string;
  customer: string;
  date: string;
}

interface Anomaly {
  id: number;
  description: string;
  amount: number;
  balance: number;
  anomaly_score: number;
  date: string;
}

interface CustomerSummary {
  customer_id: string;
  current_balance: number;
  total_transactions: number;
  account_type: string;
  total_credits: number;
  total_debits: number;
  avg_transaction_amount: number;
  largest_credit: number;
  largest_debit: number;
}

const App: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [customers, setCustomers] = useState<string[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState<string>('');
  const [customerSummary, setCustomerSummary] = useState<CustomerSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [dateFilter, setDateFilter] = useState<string>('all');

  useEffect(() => {
    fetchCustomers();
    fetchTransactions();
  }, []);

  useEffect(() => {
    if (selectedCustomer) {
      fetchMLData(selectedCustomer);
    }
  }, [selectedCustomer]);

  const fetchCustomers = async () => {
    try {
      const response = await fetch('http://localhost:8000/customers');
      const data = await response.json();
      setCustomers(data.customers);
      if (data.customers.length > 0) {
        setSelectedCustomer(data.customers[0]);
      }
    } catch (error) {
      console.error('Error fetching customers:', error);
    }
  };

  const fetchTransactions = async () => {
    try {
      const response = await fetch('http://localhost:8000/transactions/?limit=50');
      const data = await response.json();
      setTransactions(data);
    } catch (error) {
      console.error('Error fetching transactions:', error);
    }
  };

  const fetchMLData = async (customerId: string) => {
    setLoading(true);
    try {
      const [predResponse, alertResponse, anomalyResponse, summaryResponse] = await Promise.all([
        fetch(`http://localhost:8000/predict-balance/${customerId}?days_ahead=7`),
        fetch(`http://localhost:8000/generate-alerts/${customerId}`),
        fetch(`http://localhost:8000/detect-anomalies/${customerId}`),
        fetch(`http://localhost:8000/customer-summary/${customerId}`)
      ]);

      const [predData, alertData, anomalyData, summaryData] = await Promise.all([
        predResponse.json(),
        alertResponse.json(),
        anomalyResponse.json(),
        summaryResponse.json()
      ]);

      setPredictions(predData.predictions || []);
      setAlerts(alertData.alerts || []);
      setAnomalies(anomalyData.anomalies || []);
      setCustomerSummary(summaryData);

    } catch (error) {
      console.error('Error fetching ML data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const getBalanceColor = (balance: number) => {
    if (balance < 0) return 'text-red-600';
    if (balance < 1000) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getSeverityBadge = (severity: string) => {
    const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
    switch (severity) {
      case 'high': 
        return `${baseClasses} bg-red-100 text-red-800`;
      case 'medium': 
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      default: 
        return `${baseClasses} bg-blue-100 text-blue-800`;
    }
  };

  const getAccountIcon = (accountType: string) => {
    switch (accountType.toLowerCase()) {
      case 'student': return '🎓';
      case 'highnetworth': return '💎';
      case 'business': return '🏢';
      default: return '👤';
    }
  };

  const filteredTransactions = transactions.filter(t => 
    t.customer === selectedCustomer &&
    (searchTerm === '' || 
     t.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
     t.customer.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const getSpendingInsight = () => {
    if (!customerSummary) return null;
    
    const spendingRatio = customerSummary.total_debits / customerSummary.total_transactions;
    const avgSpending = Math.abs(customerSummary.largest_debit);
    
    return {
      spendingFrequency: spendingRatio > 0.6 ? 'High' : spendingRatio > 0.3 ? 'Medium' : 'Low',
      riskLevel: customerSummary.current_balance < 0 ? 'High' : customerSummary.current_balance < 1000 ? 'Medium' : 'Low'
    };
  };

  const insight = getSpendingInsight();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
            <div className="mb-4 lg:mb-0">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Chase Predictive Analytics
              </h1>
              <p className="text-gray-600 mt-1">AI-powered financial intelligence platform</p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <select
                value={selectedCustomer}
                onChange={(e) => setSelectedCustomer(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm"
              >
                <option value="">Select Customer</option>
                {customers.map((customer) => (
                  <option key={customer} value={customer}>
                    {getAccountIcon(customer.split('_')[0])} {customer}
                  </option>
                ))}
              </select>
              
              <input
                type="text"
                placeholder="Search transactions..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
              />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-xl shadow-lg mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {['overview', 'predictions', 'alerts', 'analytics', 'transactions'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`py-4 px-2 border-b-2 font-medium text-sm capitalize transition-colors ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {loading ? (
          <div className="bg-white rounded-xl shadow-lg p-12">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-lg text-gray-600">Loading AI insights...</span>
            </div>
          </div>
        ) : (
          <>
            {/* Overview Tab */}
            {activeTab === 'overview' && customerSummary && (
              <div className="space-y-6">
                {/* Customer Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-blue-100 text-sm">Current Balance</p>
                        <p className="text-2xl font-bold">
                          {formatCurrency(customerSummary.current_balance)}
                        </p>
                      </div>
                      <div className="text-3xl">💰</div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-green-100 text-sm">Total Transactions</p>
                        <p className="text-2xl font-bold">{customerSummary.total_transactions}</p>
                      </div>
                      <div className="text-3xl">📊</div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-purple-100 text-sm">Account Type</p>
                        <p className="text-2xl font-bold capitalize">{customerSummary.account_type}</p>
                      </div>
                      <div className="text-3xl">{getAccountIcon(customerSummary.account_type)}</div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-orange-100 text-sm">Active Alerts</p>
                        <p className="text-2xl font-bold">{alerts.length}</p>
                      </div>
                      <div className="text-3xl">🚨</div>
                    </div>
                  </div>
                </div>

                {/* Insights Panel */}
                {insight && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-semibold mb-4">Account Insights</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-600">Spending Frequency</p>
                        <p className={`text-lg font-semibold ${
                          insight.spendingFrequency === 'High' ? 'text-red-600' : 
                          insight.spendingFrequency === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          {insight.spendingFrequency}
                        </p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-600">Risk Level</p>
                        <p className={`text-lg font-semibold ${
                          insight.riskLevel === 'High' ? 'text-red-600' : 
                          insight.riskLevel === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          {insight.riskLevel}
                        </p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-600">Avg Transaction</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {formatCurrency(customerSummary.avg_transaction_amount)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Predictions Tab */}
            {activeTab === 'predictions' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-6">7-Day Balance Forecast</h3>
                {predictions.length > 0 ? (
                  <div className="space-y-4">
                    {predictions.map((pred, index) => {
                      const prevBalance = index === 0 
                        ? customerSummary?.current_balance || 0
                        : predictions[index - 1].predicted_balance;
                      const change = pred.predicted_balance - prevBalance;
                      
                      return (
                        <div key={pred.day} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-4">
                            <div className="text-2xl">
                              {index === 0 ? '📅' : index === 6 ? '🎯' : '📈'}
                            </div>
                            <div>
                              <p className="font-medium">Day {pred.day}</p>
                              <p className="text-sm text-gray-600">
                                {new Date(pred.date).toLocaleDateString('en-US', { 
                                  weekday: 'long', 
                                  month: 'short', 
                                  day: 'numeric' 
                                })}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className={`text-lg font-semibold ${getBalanceColor(pred.predicted_balance)}`}>
                              {formatCurrency(pred.predicted_balance)}
                            </p>
                            <p className={`text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {change >= 0 ? '↗' : '↘'} {formatCurrency(Math.abs(change))}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">Select a customer to view predictions</p>
                )}
              </div>
            )}

            {/* Alerts Tab */}
            {activeTab === 'alerts' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-6">Smart Alerts</h3>
                {alerts.length > 0 ? (
                  <div className="space-y-4">
                    {alerts.map((alert, index) => (
                      <div key={index} className="border-l-4 border-red-400 bg-red-50 p-4 rounded-r-lg">
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-3">
                            <div className="text-2xl">
                              {alert.severity === 'high' ? '🚨' : 
                               alert.severity === 'medium' ? '⚠️' : 'ℹ️'}
                            </div>
                            <div>
                              <div className="flex items-center space-x-2 mb-2">
                                <span className={getSeverityBadge(alert.severity)}>
                                  {alert.severity.toUpperCase()}
                                </span>
                                <span className="text-xs text-gray-500">
                                  {alert.type.replace('_', ' ').toUpperCase()}
                                </span>
                              </div>
                              <p className="text-gray-900 font-medium">{alert.message}</p>
                              <p className="text-xs text-gray-500 mt-1">
                                {new Date(alert.date).toLocaleString()}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="text-4xl mb-2">✅</div>
                    <p className="text-gray-500">No alerts for this customer</p>
                  </div>
                )}
              </div>
            )}

            {/* Analytics Tab */}
            {activeTab === 'analytics' && (
              <div className="space-y-6">
                {anomalies.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-semibold mb-6">Anomaly Detection</h3>
                    <div className="space-y-4">
                      {anomalies.map((anomaly, index) => (
                        <div key={index} className="border border-orange-200 rounded-lg p-4 bg-orange-50">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="text-2xl">🔍</div>
                              <div>
                                <p className="font-medium text-gray-900">{anomaly.description}</p>
                                <p className="text-sm text-gray-600">
                                  Amount: {formatCurrency(anomaly.amount)} • 
                                  Score: {anomaly.anomaly_score.toFixed(2)} • 
                                  Balance: {formatCurrency(anomaly.balance)}
                                </p>
                              </div>
                            </div>
                            <span className="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm font-medium">
                              Unusual
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {customerSummary && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-semibold mb-6">Spending Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                          <span className="text-green-700">Credits</span>
                          <span className="font-semibold text-green-800">{customerSummary.total_credits}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                          <span className="text-red-700">Debits</span>
                          <span className="font-semibold text-red-800">{customerSummary.total_debits}</span>
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                          <span className="text-blue-700">Largest Credit</span>
                          <span className="font-semibold text-blue-800">
                            {formatCurrency(customerSummary.largest_credit)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                          <span className="text-purple-700">Largest Debit</span>
                          <span className="font-semibold text-purple-800">
                            {formatCurrency(Math.abs(customerSummary.largest_debit))}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Transactions Tab */}
            {activeTab === 'transactions' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-6">Transaction History</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b-2 border-gray-200">
                        <th className="text-left py-3 px-2">Date</th>
                        <th className="text-left py-3 px-2">Description</th>
                        <th className="text-right py-3 px-2">Amount</th>
                        <th className="text-right py-3 px-2">Balance</th>
                        <th className="text-center py-3 px-2">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTransactions.slice(0, 15).map((transaction) => (
                        <tr key={transaction.id} className="border-b hover:bg-gray-50 transition-colors">
                          <td className="py-3 px-2 text-sm text-gray-600">
                            {new Date(transaction.date).toLocaleDateString()}
                          </td>
                          <td className="py-3 px-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-sm">
                                {transaction.description.includes('Bill') ? '💳' : 
                                 transaction.description.includes('Purchase') ? '🛍️' : 
                                 transaction.amount > 0 ? '💰' : '💸'}
                              </span>
                              <span>{transaction.description}</span>
                            </div>
                          </td>
                          <td className={`py-3 px-2 text-right font-medium ${
                            transaction.amount >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {formatCurrency(transaction.amount)}
                          </td>
                          <td className={`py-3 px-2 text-right font-medium ${getBalanceColor(transaction.balance)}`}>
                            {formatCurrency(transaction.balance)}
                          </td>
                          <td className="py-3 px-2 text-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              transaction.amount >= 0 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {transaction.amount >= 0 ? 'Credit' : 'Debit'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default App;