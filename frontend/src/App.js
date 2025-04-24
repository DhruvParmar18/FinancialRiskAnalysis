import React, { useState, useEffect } from 'react';
import VaRForm from './components/VaRForm';
import VaRResult from './components/VaRResult';
import './App.css'; // You can create an App.css for styling

function App() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [training, setTraining] = useState(false);
    const [trainingMessage, setTrainingMessage] = useState('');
    const [ticker, setTicker] = useState(''); // Track the ticker
    const [amount, setAmount] = useState(''); // Track the investment amount

    const trainModel = async (tickerToTrain) => {
        setTraining(true);
        setTrainingMessage('Training model...');
        setError(null);
        try {
            const response = await fetch('/train/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker: tickerToTrain }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to train model');
            }
            setTrainingMessage(data.message || 'Model training completed');
            setTicker(tickerToTrain); // Store the trained ticker
        } catch (error) {
            console.error('Training Error:', error);
            setError(error.message || 'Failed to train model');
        } finally {
            setTraining(false);
        }
    };

    const calculateVaR = async (data) => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }
            const resultData = await response.json();
            setResult(resultData);
        } catch (error) {
            console.error('Error:', error);
            setError(error.message || 'Could not calculate VaR/CVaR');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <h1>Financial Risk Analysis</h1>

            <div className="train-section">
                <h2>Train Model</h2>
                {trainingMessage && <p className="training-message">{trainingMessage}</p>}
                {error && <p className="error">{error}</p>}
                <input
                    type="text"
                    placeholder="Enter ticker to train (e.g., AAPL)"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                />
                <button onClick={() => trainModel(ticker)} disabled={training}>
                    {training ? 'Training...' : 'Train Model'}
                </button>
            </div>

            <div className="calculate-section">
                <h2>Calculate VaR/CVaR</h2>
                <VaRForm onCalculate={(data) => {
                    setAmount(data.amount); // Store the amount
                    calculateVaR({ ...data, ticker: ticker }); // Use the trained ticker
                }} />
                {loading && <div className="loading">Calculating...</div>}
                {error && <div className="error">Error: {error}</div>}
                {result && <VaRResult result={result} />}
            </div>
        </div>
    );
}

export default App;