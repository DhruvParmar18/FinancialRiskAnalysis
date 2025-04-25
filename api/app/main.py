from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from train import train_model, create_features, preprocess_data, fetch_data, fetch_macro_data, macro_indicators, merge_df
from typing import Dict, Any
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = FastAPI()

MODEL_DIR = "/usr/src/app/models"

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
    logger.info("Received training request for %s", request.ticker)
    try:
        result: Dict[str, Any] = train_model(request.ticker)
        if "error" in result:
            logger.error("Training failed for %s: %s", request.ticker, result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
        logger.info("Training succeeded for %s (model saved to %s)", request.ticker, result["model_path"])
        return TrainResponse(**result)
    except Exception as e:
        logger.exception("Unhandled exception during training for %s", request.ticker)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        ticker = request.ticker
        investment_amount = request.investment_amount
        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_model.keras")

        # Load the trained model and scaler
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

        # Fetch data
        stock_data = fetch_data([ticker])[ticker]
        macro_data = fetch_macro_data(macro_indicators)
        merged_df = merge_df(stock_data, macro_data)
        full_df = preprocess_data(create_features(merged_df, ticker))

        # Prepare input for prediction
        input_df = full_df.drop(columns=[f'{ticker}_LogReturn'])
        input_reshaped = np.array(input_df,dtype=np.float32).reshape((input_df.shape[0], input_df.shape[1], 1))
        pred = model.predict(input_reshaped)

        # Calculate VaR and CVaR from simulated returns
        var_percentage = compute_var(np.array(pred), confidence=0.95)
        cvar_percentage = compute_cvar(np.array(pred), confidence=0.95)

        var_value = investment_amount * var_percentage
        cvar_value = investment_amount * cvar_percentage

        return PredictionResponse(
            var_percentage=float(var_percentage),
            var_value=float(var_value),
            cvar_percentage=float(cvar_percentage),
            cvar_value=float(cvar_value)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Uvicorn in standalone mode")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)