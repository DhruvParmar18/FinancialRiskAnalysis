from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
import joblib
import os
from typing import Dict, Any

app = FastAPI()

MODEL_DIR = "/app/models"  # Consistent model directory


# Import the training script
try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training')))
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


class PredictionResponse(BaseModel):
    var_percentage: float
    var_value: float
    cvar_percentage: float
    cvar_value: float


def compute_var(returns, confidence=0.95):
    # Directly using the compute_var function from the provided snippet
    return np.percentile(returns, (1 - confidence) * 100)


def compute_cvar(returns, confidence=0.95):
    # Directly using the compute_cvar function from the provided snippet
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
        stock_data['LogReturn'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data = stock_data.dropna()

        # Preprocess data (Crucially, use the SAME scaler!)
        # Assuming the last 60 log returns are used as input to the model
        if len(stock_data) < 60:
            raise HTTPException(status_code=400, detail="Not enough historical data to make a prediction.")
        X = stock_data['LogReturn'].values[-60:].reshape(1, 60, 1)
        X_scaled = scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).reshape(X.shape)

        prediction = model.predict(X_scaled)

        # Invert scaling (if you scaled the target)
        # Assuming you scaled the target variable (log return) in training
        prediction_original_scale = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

        # Calculate VaR and CVaR using the historical log returns
        log_returns = stock_data['LogReturn'].dropna()
        var_percentage = compute_var(log_returns, confidence=0.95)
        cvar_percentage = compute_cvar(log_returns, confidence=0.95)

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