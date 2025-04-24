from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Dict, Any

app = FastAPI()

MODEL_DIR = "/app/models"


# Import the training script
try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "training")))
    import train
except ImportError as e:
    print(f"Error importing training script: {e}")
    raise


class TrainRequest(BaseModel):
    ticker: str


class TrainResponse(BaseModel):
    message: str
    model_path: str = None
    scaler_path: str = None
    error: str = None


class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    simulation_count: int = 1000  # Number of simulation paths


class PredictionResponse(BaseModel):
    var_percentage: float
    var_value: float
    cvar_percentage: float
    cvar_value: float


def compute_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)


def compute_cvar(returns, confidence=0.95):
    var = compute_var(returns, confidence)
    return returns[returns < var].mean()


@app.post("/train/", response_model=TrainResponse)
async def train_model_endpoint(request: TrainRequest):
    try:
        result: Dict[str, Any] = train.train_model(request.ticker)  # Call the training function
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return TrainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        ticker = request.ticker
        investment_amount = request.investment_amount
        simulation_count = request.simulation_count  # Get simulation count

        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_model.h5")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")

        # Load the trained model and scaler
        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model or scaler: {e}")

        # Fetch data
        stock_data = yf.download(ticker, period="max")
        stock_data["LogReturn"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
        stock_data = stock_data.dropna()

        # Prepare input for prediction
        if len(stock_data) < 60:
            raise HTTPException(status_code=400, detail="Not enough historical data to make a prediction.")
        X_last_60_days = stock_data["LogReturn"].values[-60:].reshape(1, 60, 1)
        X_scaled = scaler.transform(X_last_60_days.reshape(X_last_60_days.shape[0] * X_last_60_days.shape[1], X_last_60_days.shape[2])).reshape(X_last_60_days.shape)

        # Generate simulated returns
        simulated_returns = []
        for _ in range(simulation_count):
            predicted_return = model.predict(X_scaled)[0][0]
            #   Assuming we use the predicted return directly as a possible future return
            simulated_returns.append(predicted_return)

        # Calculate VaR and CVaR from simulated returns
        var_percentage = compute_var(np.array(simulated_returns), confidence=0.95)
        cvar_percentage = compute_cvar(np.array(simulated_returns), confidence=0.95)

        var_value = investment_amount * var_percentage
        cvar_value = investment_amount * cvar_percentage

        return PredictionResponse(
            var_percentage=float(var_percentage),
            var_value=float(var_value),
            cvar_percentage=float(cvar_percentage),
            cvar_value=float(cvar_value)
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))