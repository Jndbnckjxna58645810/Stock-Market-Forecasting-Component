# Stock Market Forecasting System

A modular machine learning-based system for forecasting financial time series using technical indicators, macroeconomic data, ensemble learning algorithms, and recurrent neural networks.

The project was developed as part of a bachelor’s thesis focused on the design and implementation of software components for stock market forecasting.

# Features

- Financial time series forecasting
- Support for multiple machine learning models:
  - Random Forest
  - XGBoost
  - LSTM neural networks
- Technical indicator generation
- Macroeconomic feature integration
- Configurable forecasting horizons
- Interactive Streamlit-based user interface
- Training, prediction, and evaluation pipelines
- Automatic metric calculation and visualization
- Model and configuration serialization
- Dataset caching and preprocessing utilities

# Technology Stack

- Python 3.11
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Streamlit
- Plotly
- yfinance
- fredapi
- python-dotenv

# Project Structure

project_root/

├── .env  
├── .gitignore  
├── README.md  
├── requirements.txt  
├── setup.py  
├── config/                  # Configuration files
├── data/                    # Cached datasets and generated artifacts  
├── models/                  # Saved trained models  
└── src/  
    ├── config/ 
    ├── data/                # Data loading and preprocessing  
    ├── features/            # Feature & Target engineering  
    ├── models/              # Model implementations  
    ├── pipeline/            # Dataset generation  
    ├── settings/            # Configuration 
    ├── ui/                  # Streamlit pages
    └──pages/               # Streamlit pages    

# Installation

Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```

Create a virtual environment:
```bash
python -m venv venv
```
Activate the environment.

Windows:
```bash
venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Create a .env file in the project root directory.

Example:
```bash
NEWS_API_KEY=your_key_here
FRED_API_KEY=your_key_here
HF_TOKEN=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

Start the Streamlit application:
```bash
streamlit run app.py
```
The application will open automatically in a local browser window.

# Supported Forecasting Models

The system currently supports:

## Random Forest

Ensemble model based on decision trees.

Suitable for:
- tabular feature-based forecasting;
- nonlinear dependencies;
- stable baseline experiments.

## XGBoost

Gradient boosting framework optimized for structured financial data.

Advantages:
- fast training;
- strong performance on engineered features;
- efficient handling of nonlinear interactions.

## LSTM

Recurrent neural network architecture for sequential time series modeling.

Advantages:
- temporal dependency modeling;
- sequence-based forecasting;
- internal memory mechanism for historical context.

# Supported Forecasting Targets

The system supports:

- Return forecasting
- Direction forecasting
- Price forecasting

Main experiments were focused on return prediction for:
- 1-day horizon;
- 3-day horizon;
- 5-day horizon.

# Feature Engineering

The system supports multiple groups of features.

## Trend Indicators

- SMA
- EMA
- MACD
- MACD Histogram
- Bollinger Bands
- Distance to SMA

## Momentum and Volatility Indicators

- Momentum
- RSI
- Volatility
- Volatility Ratio
- Z-score normalization

## Lag Features

- Price lag values
- Return lag values

## Volume and Candlestick Features

- Volume SMA
- Volume Ratio
- Candle body
- High-low range
- Relative candle position

## Breakout Features

- Rolling maximum/minimum
- Breakout detection

## Macroeconomic Features

- Interest Rate
- Inflation (CPI)
- Unemployment Rate
- GDP
- VIX volatility index

# Model Storage

Different serialization formats are used depending on the model type.

| Model Type | Format |
|---|---|
| Random Forest | `.pkl` |
| XGBoost | `.pkl` |
| LSTM | `.keras` |
| Configurations | `.json` |

# Evaluation Metrics

Implemented evaluation metrics include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score
- Directional Accuracy
- Baseline comparison metrics

# Notes

- The system is intended primarily for research and educational purposes.
- Financial forecasting remains a highly stochastic problem.
- Results may vary depending on:
  - selected asset;
  - interval;
  - feature configuration;
  - market conditions.
- TensorFlow is used in CPU mode by default.
- GPU acceleration is optional and was not required for the conducted experiments.

# Reproducibility

The repository contains the finalized implementation used for the experimental section of the thesis.

Experimental results may vary slightly due to:
- stochastic optimization;
- random initialization;
- external data updates.

For reproducibility, fixed random seeds are recommended where applicable.

# Future Improvements

Possible future extensions include:

- Transformer-based architectures
- Attention mechanisms
- Real-time forecasting
- Portfolio optimization
- Hyperparameter optimization
- GPU acceleration
- Additional financial instruments
- Advanced sentiment analysis

# `.gitignore` Recommendation

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
models/
storage/cache/
```

# Requirements

Example `requirements.txt`:

```bash
streamlit
pandas
numpy
scikit-learn
tensorflow
xgboost
plotly
yfinance
fredapi
joblib
tenacity
python-dotenv
```
# License

This project is distributed for educational and research purposes only.

No investment advice is provided.