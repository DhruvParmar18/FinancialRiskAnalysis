import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
import os  # For creating directories


macro_indicators = ["GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "GS10", "INDPRO", "PPIACO", "RSXFS", "HOUST", "PSAVERT"]
MODEL_DIR = "/app/models"  # Consistent model directory


def fetch_data(asset_list):
    data = {}
    for asset in asset_list:
        stock = yf.Ticker(asset)
        data[asset] = stock.history(period="max")[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data


def fetch_macro_data(indicators):
    start = datetime.datetime(1992, 1, 1)
    end = datetime.datetime.now()
    return {ind: web.DataReader(ind, 'fred', start, end) for ind in indicators}


def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    return df


def create_features(df, asset_name):
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
    features = full_df.columns.difference([f'{asset_name}_LogReturn'])
    X = full_df[features].values
    y = full_df[f'{asset_name}_LogReturn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler  # Return the scaler


def build_lstm_model(input_shape):
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


def arima_baseline(y_train, y_test):
    model = ARIMA(y_train, order=(5, 1, 0)).fit()
    return model.forecast(steps=len(y_test))


def linear_regression_baseline(X_train, X_test, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_model(y_true, y_pred):
    print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_true, y_pred))
    print("R-Squared Score:", r2_score(y_true, y_pred))


def backtest_strategy(y_test, y_pred, asset_name):
    if isinstance(y_pred, pd.Series):
        y_pred_series = y_pred
    else:
        y_pred_series = pd.Series(y_pred.flatten(), index=y_test.index)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_pred_series.index, y_pred_series, label='Predicted', color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.title(f"{asset_name} Model Backtest Performance")
    plt.legend()
    plt.show()


def train_model(ticker: str):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure model directory exists

        stock_data = fetch_data([ticker])[ticker]
        macro_data = fetch_macro_data(macro_indicators)
        full_df = preprocess_data(create_features(stock_data, ticker))

        X_train, X_test, y_train, y_test, scaler = train_test_split_data(
            full_df.drop(columns=[f'{ticker}_LogReturn']), full_df[f'{ticker}_LogReturn'], ticker)  # Get the scaler

        input_shape = (X_train.shape[1], 1)
        model = build_lstm_model(input_shape)

        X_train_reshaped = np.array(X_train, dtype=np.float32).reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = np.array(X_test, dtype=np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))

        history = model.fit(
            X_train_reshaped, np.array(y_train, dtype=np.float32),
            epochs=50, batch_size=32, validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
        )

        y_pred = model.predict(X_test_reshaped)
        arima_pred = arima_baseline(y_train, y_test)
        lr_pred = linear_regression_baseline(X_train, X_test, y_train)

        print(f"Comparison for {ticker}:")
        print("LSTM Model:")
        evaluate_model(y_test, y_pred)
        print("ARIMA Model:")
        evaluate_model(y_test, arima_pred)
        print("Linear Regression Model:")
        evaluate_model(y_test, lr_pred)

        # Save the model and the scaler with ticker-specific filenames
        model_filename = os.path.join(MODEL_DIR, f"{ticker}_lstm_model.h5")
        scaler_filename = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
        model.save(model_filename)
        import joblib
        joblib.dump(scaler, scaler_filename)

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

        return {"message": f"Training completed successfully for {ticker}", "model_path": model_filename, "scaler_path": scaler_filename}

    except Exception as e:
        return {"error": str(e)}