import React from 'react';

function VaRResult({ result }) {
  if (!result) return null;

  return (
    <div>
      <h2>VaR/CVaR Results</h2>
      <p>VaR (95%): {result.var_percentage.toFixed(4)} ({result.var_value.toFixed(2)})</p>
      <p>CVaR (95%): {result.cvar_percentage.toFixed(4)} ({result.cvar_value.toFixed(2)})</p>
    </div>
  );
}

export default VaRResult;