"""
Financial Forecasting with Deep Learning Models
------------------------------------------------

This script implements and trains several deep neural network architectures
for predicting future stock prices from historical data.  It is designed to
accompany the `financial-forecasting-cnn-lstm` GitHub repository and serves as a
reference implementation for the models described in the coursework report.

Key features:

* Reads a CSV file containing daily stock prices with at least a `Date`
  column and one numerical feature (e.g. `Open`, `Close` or `Adj Close`).
* Performs basic preprocessing: converts dates to pandas `DatetimeIndex`,
  sorts the data chronologically and scales the numerical column between 0
  and 1 using a `MinMaxScaler`.
* Constructs input/output sequences using a sliding window (default 30 days).
* Splits the sequences into training, validation and test sets (70/15/15
  proportion by default).
* Defines four neural network models using TensorFlow/Keras:
    - Long Short‑Term Memory (LSTM)
    - Gated Recurrent Unit (GRU)
    - Deep Regression (CNN‑LSTM hybrid)
    - Sequential Dense neural network
* Compiles each model with the Adam optimiser and mean‑squared error loss
  and trains them with early stopping to prevent overfitting.
* Prints performance metrics (MAE, MSE, RMSE, R², MAPE) on the test set.

The code is deliberately modular so that additional models or preprocessing
steps can be incorporated easily.  To use, update the `DATA_PATH` variable
with the path to your CSV file and run the script:

```sh
python financial_forecasting.py
```

Dependencies:

* pandas
* numpy
* scikit‑learn
* matplotlib (optional for plotting)
* tensorflow

Note: training deep learning models can be computationally intensive.  Adjust
the `epochs` and `batch_size` parameters as necessary for your hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for this script. Install it via pip: pip install tensorflow"
    ) from e


@dataclass
class DatasetConfig:
    """Configuration for dataset and sequence generation."""
    data_path: str
    column_name: str = "Open"  # Name of the numerical column to forecast
    lookback: int = 30  # Number of past days used for forecasting
    train_split: float = 0.7  # Proportion of sequences used for training
    val_split: float = 0.15  # Proportion used for validation; rest is test


def load_and_preprocess(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Load the CSV file, scale the numerical column and return values array.

    Parameters
    ----------
    config: DatasetConfig
        Settings containing the path to the CSV and column name.

    Returns
    -------
    values: np.ndarray
        Scaled array of the target column.
    dates: np.ndarray
        Corresponding datetime index (useful for plotting).
    scaler: MinMaxScaler
        Fitted scaler for inverse transformation.
    """
    df = pd.read_csv(config.data_path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    if config.column_name not in df.columns:
        raise ValueError(f"Column '{config.column_name}' not found in CSV.")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    values = df[config.column_name].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    return scaled_values, df["Date"].values, scaler


def create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input/output sequences for time‑series forecasting.

    Each input sequence contains `lookback` timesteps and the target is the
    immediately following value.  For example, with lookback=30 the model sees
    values at t‑29,…,t and predicts value at t+1.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float,
    val_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into training, validation and test sets."""
    total = len(X)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def reshape_for_rnn(X: np.ndarray) -> np.ndarray:
    """Reshape 2D sequence array (samples, timesteps) into 3D shape for Keras."""
    return X.reshape((X.shape[0], X.shape[1], 1))


def build_lstm(input_shape: Tuple[int, int]) -> Sequential:
    """Create an LSTM model."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape: Tuple[int, int]) -> Sequential:
    """Create a GRU model."""
    model = Sequential(
        [
            Input(shape=input_shape),
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape: Tuple[int, int]) -> Sequential:
    """Create a CNN‑LSTM hybrid model (Deep Regression)."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_sequential_dense(input_shape: Tuple[int, int]) -> Sequential:
    """Create a Sequential Dense model treating each timestep independently."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Flatten(),
            Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model and return a dictionary of metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    # avoid divide by zero in MAPE
    mape = np.mean(np.abs((y_test - preds) / np.where(y_test == 0, 1e-8, y_test))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}


def main():
    # Adjust this path to point to your CSV file
    DATA_PATH = os.environ.get("STOCK_DATA_PATH", "data/starbucks_stock.csv")
    config = DatasetConfig(data_path=DATA_PATH, column_name="Open", lookback=30)

    data, dates, scaler = load_and_preprocess(config)
    X, y = create_sequences(data, config.lookback)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, config.train_split, config.val_split)
    # reshape for RNN models
    X_train_rnn = reshape_for_rnn(X_train)
    X_val_rnn = reshape_for_rnn(X_val)
    X_test_rnn = reshape_for_rnn(X_test)

    models = {
        "LSTM": build_lstm((config.lookback, 1)),
        "GRU": build_gru((config.lookback, 1)),
        "DeepRegression": build_cnn_lstm((config.lookback, 1)),
        "SequentialDense": build_sequential_dense((config.lookback, 1)),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model…")
        # Choose appropriate training data shape (flattened vs 3D)
        X_train_input = X_train_rnn if name != "SequentialDense" else X_train
        X_val_input = X_val_rnn if name != "SequentialDense" else X_val
        X_test_input = X_test_rnn if name != "SequentialDense" else X_test
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(
            X_train_input,
            y_train,
            validation_data=(X_val_input, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )
        metrics = evaluate_model(model, X_test_input, y_test)
        results[name] = metrics
        print(f"{name} performance: {metrics}")

    print("\nSummary of model performances:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
