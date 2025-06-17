# 📊 Stock Price Prediction App

Welcome to the **Stock Price Prediction App**, an interactive web application designed to forecast future stock prices using advanced time-series modeling techniques. This application leverages an intuitive interface, real-time data retrieval, and an AI-based forecasting library to provide quick and accessible insights for users interested in analyzing and predicting stock market behavior.

![](https://raw.github.com/alphatechlogics/StockMarketPricePrediction/8513341a5e7e9f67c62087c10dfb6ad7d7fb6953/Screenshot%202025-06-17%20222312.png)

---

## ✨ Features

- **Single-Ticker Forecasting**: Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT) to retrieve and analyze historical data.
- **Interactive Plots**: Visualize historical Open/Close prices using dynamic and responsive charts.
- **Prophet-Based Forecasting**: Use a robust forecasting library that automatically handles trends and seasonality.
- **Customizable Horizon**: Enter how many years you want to forecast, offering flexibility for short-term or mid-term predictions.
- **Forecast Components**: View trend, weekly, and yearly seasonal insights (when relevant) in a user-friendly format.
- **Evaluation Metrics**: (Optional in certain builds) Access mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), mean absolute percentage error (MAPE), and an approximate accuracy percentage for performance analysis.
- **Multi-Index Handling**: Automatically flattens multi-level data returned by Yahoo Finance for seamless processing.

---

## 💻 Tech Stack

- **Python**: Primary programming language for data manipulation and modeling.
- **Streamlit**: Framework for creating interactive web apps directly from Python scripts.
- **Prophet**: A powerful forecasting library from Meta, suitable for time series data with strong trends and seasonalities.
- **Plotly**: Provides rich, interactive graphing tools for in-app visualizations.
- **yfinance**: Downloads historical stock data from Yahoo Finance.
- **pandas**: Handles data manipulation tasks.
- **scikit-learn**: Offers metrics for model performance evaluation (MAE, MSE, RMSE, MAPE) when a test set is used.
- **NumPy**: Utilized for efficient numerical operations.

---

## 🚀 Getting Started

1. **Install Python 3.10+** (or a compatible version) if it’s not already on your system.
2. **Create (Optional) and Activate a Virtual Environment** to isolate your project’s dependencies.
3. **Install Required Libraries**: Make sure you have all the necessary packages such as Streamlit, Prophet, Plotly, scikit-learn, yfinance, pandas, and NumPy.

---

## 📥 **Installation**

Follow these steps to set up StockMarketPricePrediction on your local machine:

### 1. **Clone the Repository**

```bash
git clone https://github.com/alphatechlogics/StockMarketPricePrediction.git
cd StockMarketPricePrediction
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: Ensure that FFmpeg is installed on your system.

## 📝 Usage

1. **Open your terminal or command prompt** in the project directory where your application script is located.
2. **Run the application** by starting a local web server for Streamlit (e.g., `streamlit run app.py`).
3. **Navigate to the provided URL** (often http://localhost:8501) in your web browser.
4. **Interact with the App**:
   - Go to **“Stock Prediction”** in the sidebar.
   - Enter a valid stock symbol.
   - Select how many years of data to forecast.
   - Wait for data to load, then review:
     - Raw historical data (last few rows for a quick glance).
     - Historical time series plots.
     - Forecast results and visual forecast components.
     - Prediction vs. Actual for the selected horizon.
     - Accuracy metrics (MAE, MSE, RMSE, MAPE, and approximate accuracy).

---

## 🏦 Disclaimer

- This application is intended for **educational and informational** purposes only.
- It is **not** financial or investment advice. Markets can be volatile; always do thorough research or consult a professional financial advisor before making important investment decisions.

---
