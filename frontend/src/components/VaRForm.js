import React, { useState } from 'react';

function VaRForm({ onCalculate }) {
  const [ticker, setTicker] = useState('');
  const [amount, setAmount] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onCalculate({ ticker, amount: parseFloat(amount) });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Ticker:
        <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} required />
      </label>
      <br />
      <label>
        Investment Amount:
        <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} required />
      </label>
      <br />
      <button type="submit">Calculate VaR/CVaR</button>
    </form>
  );
}

export default VaRForm;