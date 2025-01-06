import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

selected_stock = st.text_input("Enter Stock Symbol :")

if not selected_stock:
    st.warning("Please enter a stock symbol to proceed.")
else:
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        """Fetch historical data for the given stock ticker and handle multi-index columns."""
        try:
            # 1) Download data with actions=False to avoid dividends/splits
            raw_data = yf.download(
                ticker,
                start=START,
                end=TODAY,
                actions=False
            )

            # 2) Check if the DataFrame is empty
            if raw_data.empty:
                st.error(f"No data returned for {ticker} in the specified date range.")
                return pd.DataFrame()

            # 3) If columns are MultiIndex, flatten them
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [col[0] for col in raw_data.columns]

            # 4) If 'Adj Close' is missing, create it from 'Close'
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in raw_data.columns:
                    raise ValueError(f"Missing required column '{col}' in data for {ticker}.")

            if 'Adj Close' not in raw_data.columns:
                raw_data['Adj Close'] = raw_data['Close']

            # 5) Select only the columns we need
            data = raw_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

            # 6) Reset index to make 'Date' a column
            data.reset_index(inplace=True)
            data.rename(columns={'index': 'Date'}, inplace=True)  # Just in case

            return data

        except Exception as e:
            st.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()

    # Load data and update status
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    if data.empty:
        st.error("No data found for the entered stock symbol.")
    else:
        # Raw Data
        st.subheader('Raw Data')
        st.markdown(
            """
            Below is the raw stock data for the selected ticker, from the specified start date
            to the current date. This data includes the **open**, **high**, **low**, **close**,
            **adjusted close**, and **volume** for each trading day. By examining these values,
            you can get an initial sense of how the stock has performed historically.
            """
        )
        st.write(data.tail())

        # Time Series Chart
        def plot_raw_data():
            st.markdown("#### Time Series Chart of Stock Prices")
            st.markdown(
                """
                This chart displays the historical **Open** and **Close** prices of the stock 
                over time. The **Open** price is recorded at the start of each trading day, 
                while the **Close** price is recorded at the end of each trading day. 
                Observing how these two metrics change daily can help you identify trends or 
                patterns, such as bullish or bearish runs, volatility, and potential market 
                reactions to external events.
                """
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
            fig.update_layout(
                title_text="Time Series Data",
                xaxis_rangeslider_visible=True,
                xaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=0.75),
                yaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=0.5),
                width=800,
                height=600
            )
            st.plotly_chart(fig)

        plot_raw_data()

        # Prepare data for Prophet: rename columns for Prophet's expected format (ds, y)
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # Split into train and test
        train_data = df_train[:-period]
        test_data = df_train[-period:]

        # Prophet model
        m = Prophet()
        m.fit(train_data)

        # Forecast
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Forecast Data
        st.subheader('Forecast Data')
        st.markdown(
            """
            Below is a preview of the **forecasted stock prices** generated by the Prophet model. 
            The key columns include:
            - **ds** (date)
            - **yhat** (predicted value of the stock's closing price)
            - **yhat_lower** and **yhat_upper** (lower and upper bounds of the prediction interval).
            
            These bounds can help you understand the level of uncertainty in the model’s predictions.
            """
        )
        st.write(forecast.tail())

        # Forecast Chart
        st.markdown("#### Forecast Chart")
        st.markdown(
            """
            The chart below overlays the **historical data** and the **forecast** into the future. 
            The darker line represents the predicted trend, while the lighter blue areas 
            indicate the model's uncertainty intervals. The vertical line often marks the 
            boundary between the historical data and the forecasted period.
            """
        )
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        # Forecast Components
        st.markdown("#### Forecast Components")
        st.markdown(
            """
            These plots break down the forecast into its main components, including:
            - **Trend**: Long-term direction of the stock price.
            - **Yearly Seasonality**: Patterns that repeat every year (e.g., seasonal business cycles).
            - **Weekly Seasonality**: Patterns that repeat every week (e.g., higher trading volumes on certain weekdays).
            By analyzing each component, you can better understand the factors that contribute to 
            the overall forecast.
            """
        )
        fig2 = m.plot_components(forecast)
        st.write(fig2)

        # Prediction vs Actual
        test_pred = forecast[['ds', 'yhat']].tail(period)
        test_actual = test_data.reset_index(drop=True)
        comparison = pd.merge(test_pred, test_actual, how='inner', on='ds')

        st.subheader('Prediction vs Actual')
        st.markdown(
            """
            This table compares the model's predicted values (`yhat`) with the **actual stock 
            closing values** (`y`) for the test period. By examining these two columns side by side, 
            you can evaluate how closely the model's predictions match real market outcomes. 
            Significant differences may point to sudden market changes or external factors that 
            the model didn't anticipate.
            """
        )
        st.write(comparison)

        # Metrics
        mae = mean_absolute_error(comparison['y'], comparison['yhat'])
        mse = mean_squared_error(comparison['y'], comparison['yhat'])
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(comparison['y'], comparison['yhat'])

        avg_prediction = comparison['yhat'].mean()
        avg_actual = comparison['y'].mean()
        accuracy_percentage = 100 - (mape * 100)

        st.subheader('Accuracy Metrics')
        st.markdown(
            """
            Below are several metrics that quantify the **accuracy** of the model’s predictions:
            
            - **Mean Absolute Error (MAE)**: The average magnitude of errors in a set of predictions, 
              without considering their direction. It’s calculated as the average of absolute 
              differences between predicted and actual values.
              
            - **Mean Squared Error (MSE)**: Similar to MAE, but it squares the errors before 
              averaging them. This means **larger errors** have a bigger impact.
              
            - **Root Mean Squared Error (RMSE)**: The square root of MSE, which brings the 
              error metric back to the same **units** as the original data (i.e., stock price in this case).
              
            - **Mean Absolute Percentage Error (MAPE)**: The average of absolute percentage 
              errors between predicted and actual values, giving a sense of the size of the errors 
              in relative terms.
              
            - **Accuracy** (as defined here) = 100 - (MAPE * 100). This gives a quick view of how 
              close (in percentage) the predictions were to the actual values on average.
            """
        )

        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.4f}")
        st.write(f"**Average Prediction:** {avg_prediction:.2f}")
        st.write(f"**Average Actual:** {avg_actual:.2f}")
        st.write(f"**Accuracy:** {accuracy_percentage:.2f}%")
