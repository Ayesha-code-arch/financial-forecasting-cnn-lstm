# Financial Forecasting with Deep Learning

This repository contains code and reports for predicting Starbucks Corporation (SBUX) share prices using a range of deep‑learning models.  Daily closing prices were collected for the period **2017‑2025** and used to train recurrent neural networks (RNNs), including Long Short‑Term Memory (LSTM), Gated Recurrent Unit (GRU), a **CNN‑LSTM hybrid**, and a fully connected dense model.  The aim is to model both long‑ and short‑term temporal dependencies to forecast future values and evaluate which architecture generalises best.

## Data

The dataset comprises daily closing prices and volumes for Starbucks shares from January 2017 to January 2025.  Time‑series data were obtained from a public financial API.  A sliding‑window approach was used to convert the univariate series into supervised learning samples.

- **Features:** Each input sequence contains `n` past closing prices; the target is the next day’s closing price.  Window sizes between 30–100 days were explored.
- **Splits:** The time series was split chronologically into training, validation and test sets to avoid data leakage.  Data scaling (e.g. Min‑Max or StandardScaler) was applied based on the training set only.

## Models & Methods

Four neural‑network architectures were implemented using Keras:

1. **LSTM** – captures long‑range dependencies by controlling information flow with gates.  A single LSTM layer with 50 units was followed by dense layers.
2. **GRU** – similar to LSTM but with fewer gates and parameters, potentially speeding training.
3. **CNN‑LSTM hybrid** – a one‑dimensional convolutional layer extracts local patterns from sliding windows before passing sequences to an LSTM layer.  This combination aims to learn both local and global patterns.
4. **Dense (feed‑forward) model** – baseline sequential network treating the windowed series as a fixed‑length vector without temporal recurrence.

Hyper‑parameters (number of units, learning rate, epochs, window size) were tuned via manual search.  Early stopping monitored validation loss to prevent overfitting.  For each model we trained on the training set, evaluated on the validation set and reported performance on the test set.

## Results

Model performance was measured using the **coefficient of determination (R²)** and Mean Absolute Error (MAE) on the unseen test set.  The **CNN‑LSTM** delivered the best generalisation with an R² of approximately **0.889** and a low MAE, indicating that combining convolutional features and recurrent memory effectively captures stock trends.  The plain LSTM and GRU achieved slightly lower R² values (around 0.85), while the dense model underperformed due to its inability to model temporal dependence.

Plots compare predicted vs actual stock prices across the test horizon.  The best model tracks the overall trend and captures both short‑term fluctuations and long‑term movements.  These visualisations are included in the `plots/` directory.

## Repository Structure

```
financial‑forecasting‑cnn‑lstm/
├── README.md          # Project description
├── LICENSE            # MIT Licence
├── data/              # (Optional) Scripts to download or preprocess data
├── src/               # Python scripts and notebooks for model training and evaluation
├── plots/             # Generated plots comparing predictions and actuals
└── reports/           # Project report and documentation
```

- The notebook/`src` folder contains Python code for preparing data, defining models, and training/evaluating them.
- The `plots` folder holds figures illustrating predicted vs actual stock prices for each model.
- The `reports` folder stores the project write‑up (PDF or DOCX) summarising methods, results, and conclusions.

## Usage

To reproduce the experiments, clone this repository and install the required Python packages (TensorFlow/Keras, NumPy, Pandas, Matplotlib).  An example workflow:

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data and create sliding windows
python sr

To run the provided Python script, set an environment variable `STOCK_DATA_PATH` to the location of your CSV file (or place `starbucks_stock.csv` inside a `data/` folder). Then execute:

```bash
python financial_forecasting.py
```

The script will preprocess the data, train the four neural network models, and print performance metrics for each model.
c/prepare_data.py --input data/starbucks_prices.csv --window 60

# Train the CNN‑LSTM model
python src/train_model.py --model cnn_lstm --window 60 --epochs 50 --batch_size 32

# Evaluate on test set and generate plots
python src/evaluate_model.py --model cnn_lstm --window 60 --output_dir plots/
```

Replace `cnn_lstm` with `lstm`, `gru` or `dense` to train other models.  Training parameters (window size, number of epochs, etc.) can be customised via command‑line options.

## Licence

This project is licensed under the **MIT Licence**.  See the `LICENSE` file for details.

## Acknowledgements

This coursework was completed as part of an MSc module on deep learning for real‑time forecasting.  It draws inspiration from common time‑series forecasting techniques and demonstrates how RNN architectures can be applied to financial data.  Please cite the original financial data provider when using the dataset.
