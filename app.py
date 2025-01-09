import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np

# ----- Global Settings -----
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Page Title
st.title("Stock Prediction App")

# Ask user for stock symbol
selected_stock = st.text_input("Enter Stock Symbol :")

# Ask user for the number of years to forecast
years_input = st.text_input("Enter the number of years to forecast:", value="3")

if not selected_stock:
    st.warning("Please enter a stock symbol to proceed.")
else:
    # Validate user input for forecast years
    if not years_input:
        st.warning("Please enter the number of years for forecasting.")
        st.stop()
    try:
        n_years = int(years_input)
    except ValueError:
        st.error("Invalid input. Please enter a valid integer for the number of years.")
        st.stop()
    
    if n_years < 1:
        st.error("Number of years must be at least 1.")
        st.stop()
    
    # Convert years to days
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        """Fetch historical data for the given stock ticker using yfinance and clean it."""
        try:
            raw_data = yf.download(
                ticker,
                start=START,
                end=TODAY,
                actions=False
            )

            # Check if the DataFrame is empty
            if raw_data.empty:
                st.error(f"No data returned for {ticker} in the specified date range.")
                return pd.DataFrame()

            # Flatten multi-index columns if necessary
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [col[0] for col in raw_data.columns]

            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in raw_data.columns:
                    raise ValueError(f"Missing required column '{col}' in data for {ticker}.")

            # If 'Adj Close' is missing, create from 'Close'
            if 'Adj Close' not in raw_data.columns:
                raw_data['Adj Close'] = raw_data['Close']

            # Create a cleaner DataFrame
            data = raw_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            data.reset_index(inplace=True)  # Make 'Date' a column
            data.rename(columns={'index': 'Date'}, inplace=True)  # Just in case

            return data

        except Exception as e:
            st.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()

    # ----- Load data -----
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # If no data, stop execution
    if data.empty:
        st.error("No data found for the entered stock symbol.")
    else:
        # ----- Show Raw Data -----
        st.subheader("Raw Data")
        st.markdown(
            """
            Below is the raw stock data for the selected ticker, from the specified start date
            to the current date. This data includes:
            - **Open**: Price at market open
            - **High**: Highest price of the day
            - **Low**: Lowest price of the day
            - **Close**: Price at market close
            - **Adj Close**: Adjusted closing price
            - **Volume**: Number of shares traded
            """
        )
        st.write(data)

        # ----- Plot Time Series -----
        st.subheader("Historical Stock Prices")
        st.markdown(
            """
            This chart shows the **Open** and **Close** prices over time. Observing these trends 
            helps identify bullish or bearish runs, overall volatility, and potential market reactions.
            """
        )
        fig_raw = go.Figure()
        fig_raw.add_trace(
            go.Scatter(
                x=data['Date'], 
                y=data['Open'], 
                mode='lines',
                line=dict(color='blue'),
                name='Stock Open'
            )
        )
        fig_raw.add_trace(
            go.Scatter(
                x=data['Date'], 
                y=data['Close'], 
                mode='lines',
                line=dict(color='red'),
                name='Stock Close'
            )
        )
        fig_raw.update_layout(
            title_text="Open vs Close Price Over Time",
            xaxis_rangeslider_visible=True,
            xaxis=dict(showgrid=True, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridcolor='LightGrey'),
            width=800,
            height=600
        )
        st.plotly_chart(fig_raw)

        # ----- Prepare Data for Prophet -----
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # Train on ALL data (no hold-out)
        train_data = df_train

        # ----- Build and Train Prophet Model -----
        m = Prophet()
        m.fit(train_data)

        # ----- Forecast -----
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # ----- Forecast Data (Historical + Future) -----
        st.subheader("Forecast Data (All)")
        st.markdown(
            """
            The **Prophet** model provides a comprehensive forecast DataFrame with:
            - **ds**: Date
            - **yhat**: Predicted closing price
            - **yhat_lower** & **yhat_upper**: Confidence interval bounds
            - Additional trend & seasonal components
            """
        )
        st.write(forecast)

        # ----- Future-Only Forecast Table -----
        st.subheader("Future Forecast Table")
        st.markdown(
            """
            Below are the **newly forecasted** dates beyond the last available historical date.
            This is where the model predicts the future trajectory of the stock's closing price.
            """
        )
        last_training_date = train_data['ds'].max()
        future_only = forecast[forecast['ds'] > last_training_date]
        st.write(future_only)

        # ----- Forecast Chart (Historical + Future) -----
        st.subheader("Forecast Chart (Historical + Future)")
        st.markdown(
            """
            This interactive chart shows both the **original historical data** (in black dots) 
            and the **forecast** (blue line) with its uncertainty intervals (light blue area).
            """
        )
        fig_forecast = plot_plotly(m, forecast)
        # Enhance layout
        fig_forecast.update_layout(
            title="Prophet Forecast with Historical Data",
            xaxis=dict(showgrid=True, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridcolor='LightGrey')
        )
        st.plotly_chart(fig_forecast)

        # ----- Future-Only Forecast Chart -----
        st.subheader("Future-Only Forecast Chart")
        st.markdown(
            """
            This chart zooms in on **only the forecasted dates** to give a clearer view of
            the model's predictions and the associated confidence intervals.
            """
        )
        fig_future = go.Figure()
        fig_future.add_trace(
            go.Scatter(
                x=future_only['ds'],
                y=future_only['yhat'],
                mode='lines',
                line=dict(color='green'),
                name='Future Predictions'
            )
        )
        # Confidence interval fill
        fig_future.add_trace(
            go.Scatter(
                x=pd.concat([future_only['ds'], future_only['ds'][::-1]]),
                y=pd.concat([future_only['yhat_lower'], future_only['yhat_upper'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Prediction Interval'
            )
        )
        fig_future.update_layout(
            title_text="Future-Only Forecast",
            xaxis_rangeslider_visible=True,
            xaxis=dict(showgrid=True, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridcolor='LightGrey'),
            width=800,
            height=600
        )
        st.plotly_chart(fig_future)

        # ----- Forecast Components -----
        st.subheader("Forecast Components")
        st.markdown(
            """
            **Prophet** breaks down the forecast into major components:
            - **Trend**: The overall upward or downward movement over time
            - **Yearly Seasonality**: Repeating patterns each year
            - **Weekly Seasonality**: Repeating patterns each week
            - (Possible) **Daily Seasonality**: If enabled

            By examining these components, we can see what drives the forecast.
            """
        )
        fig_components = m.plot_components(forecast)
        st.write(fig_components)
