from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Attention, BatchNormalization, Input
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import pandas_datareader.data as web
import datetime
from statsmodels.tsa.arima.model import ARIMA  # Import inside the function
# import matplotlib.pyplot as plt
import os  # For creating directories

macro_indicators = ["GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "GS10", "INDPRO", "PPIACO", "RSXFS", "HOUST", "PSAVERT"]
MODEL_DIR = "/usr/src/app/models"  # Consistent model directory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

logger = logging.getLogger(__name__)


def fetch_data(asset_list):
    data = {}
    for asset in asset_list:
        logger.info("Fetching price history for %s", asset)
        stock = yf.Ticker(asset)
        data[asset] = stock.history(period="max")[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data


def fetch_macro_data(indicators):
    start = datetime.datetime(1992, 1, 1)
    end = datetime.datetime.now()
    logger.info("Fetching macro‑economic indicators from FRED (%s ➞ %s)", start.date(), end.date())
    return {ind: web.DataReader(ind, 'fred', start, end) for ind in indicators}


def preprocess_data(df):
    logger.debug("Pre‑processing data frame with %d rows", len(df))
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def create_features(df, asset_name):
    logger.debug("Creating engineered features for %s", asset_name)
    df[f'{asset_name}_LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df[f'{asset_name}_RollingVolatility'] = df[f'{asset_name}_LogReturn'].rolling(window=30).std()
    df[f'{asset_name}_MaxDrawdown'] = df['Close'] / df['Close'].rolling(window=30).max() - 1
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Week'] = df.index.isocalendar().week
    df['DayOfWeek'] = df.index.dayofweek
    df[f'{asset_name}_Momentum'] = df['Close'].diff(10)
    df[f'{asset_name}_RSI'] = 100 - (
            100 / (1 + df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).mean()))
    return df


def train_test_split_data(full_df, asset_name):
    logger.debug("Splitting data into train/test sets for %s", asset_name)
    features = full_df.columns.difference([f'{asset_name}_LogReturn'])
    X = full_df[features].values
    y = full_df[f'{asset_name}_LogReturn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler  # Return the scaler


def build_lstm_model(input_shape):
    logger.info(f"building model, Input Shape : {input_shape}")
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(inputs)
    attention = Attention()([lstm_out, lstm_out])
    lstm_out = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(attention)
    lstm_out = BatchNormalization()(lstm_out)
    dense_out = Dense(32, activation="relu")(lstm_out)
    dense_out = Dropout(0.3)(dense_out)
    dense_out = Dense(16, activation="relu")(dense_out)
    output = Dense(1)(dense_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model



def evaluate_model(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info("%s – MAE: %.6f | MSE: %.6f | R²: %.4f", label, mae, mse, r2)



# def backtest_strategy(y_test, y_pred, asset_name):
#     if isinstance(y_pred, pd.Series):
#         y_pred_series = y_pred
#     else:
#         y_pred_series = pd.Series(y_pred.flatten(), index=y_test.index)
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test.index, y_test, label='Actual', color='blue')
#     plt.plot(y_pred_series.index, y_pred_series, label='Predicted', color='red', linestyle='dashed')
#     plt.xlabel("Time")
#     plt.ylabel("Log Return")
#     plt.title(f"{asset_name} Model Backtest Performance")
#     plt.legend()
#     plt.show()

def merge_df(stock_data, macro_data):
    tidy_series = []
    for name, df_or_ser in macro_data.items():
        # make sure what we have is a DataFrame with one column = the macro series
        s = (df_or_ser.squeeze()  # → Series
             .rename(name)  # give it an explicit column name
             .to_frame())  # back to single-column DataFrame
        # protect against any timezone offsets so the merge is clean
        s.index = pd.to_datetime(s.index).tz_localize(None)
        tidy_series.append(s)

    macro_df = (pd.concat(tidy_series, axis=1).sort_index())
    stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
    macro_df.index = pd.to_datetime(macro_df.index)  # already naïve but be explicit
    full_idx = pd.date_range(stock_data.index.min(),
                             stock_data.index.max(),
                             freq="D")
    macro_daily = macro_df.reindex(full_idx).ffill()
    merged = stock_data.join(macro_daily, how="left")
    return merged

def train_model(ticker: str):
    try:
        logger.info("Starting training pipeline for %s", ticker)
        os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure model directory exists

        stock_data = fetch_data([ticker])[ticker]
        macro_data = fetch_macro_data(macro_indicators)
        merged_df = merge_df(stock_data,macro_data)
        full_df = preprocess_data(create_features(merged_df, ticker))

        X_train, X_test, y_train, y_test = train_test_split(full_df.drop(columns=[f'{ticker}_LogReturn']), full_df[f'{ticker}_LogReturn'], test_size=0.2, shuffle=False)

        input_shape = (X_train.shape[1], 1)
        model = build_lstm_model(input_shape)

        X_train_reshaped = np.array(X_train, dtype=np.float32).reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = np.array(X_test, dtype=np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))

        history = model.fit(
            X_train_reshaped, np.array(y_train, dtype=np.float32),
            epochs=50, batch_size=32, validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
        )
        logger.info("Finished training – best val_loss %.6f", min(history.history['val_loss']))

        y_pred = model.predict(X_test_reshaped)
        print(f"Comparison for {ticker}:")
        print("LSTM Model:")
        evaluate_model(y_test, y_pred)

        # Save the model and the scaler with ticker-specific filenames
        model_filename = os.path.join(MODEL_DIR, f"{ticker}_lstm_model.keras")
        model.save(model_filename)
        # backtest_strategy(y_test, y_pred, ticker)
        # backtest_strategy(y_test, arima_pred, f"{ticker} ARIMA")
        # backtest_strategy(y_test, lr_pred, f"{ticker} Linear Regression")

        # # Plot Training & Validation Loss
        # plt.figure(figsize=(10, 5))
        # plt.plot(history.history['loss'], label='Training Loss', color='blue')
        # plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.title("Model Training Loss Over Time")
        # plt.legend()
        # plt.show()

        return {"message": f"Training completed successfully for {ticker}", "model_path": model_filename}

    except Exception as e:
        return {"error": str(e)}