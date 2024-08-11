# File path: stock_trend_prediction_app.py

# Install required packages if not already installed
# Uncomment the next line to install required packages
# !pip install streamlit yfinance prophet plotly

import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data[['Close']].reset_index()
    stock_data.dropna(inplace=True)  # Remove any missing values
    return stock_data

# Function to make predictions
def predict_stock_trends(data, periods=30):
    df = data[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    # Apply log transformation to stabilize variance
    df['y'] = np.log(df['y'])
    
    # Initialize and fit the model
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)
    
    # Create a future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Remove weekends and non-trading days from the future dataframe
    future['day_of_week'] = future['ds'].dt.dayofweek
    future = future[future['day_of_week'] < 5]  # Keep only weekdays (0=Monday, 4=Friday)
    future.drop(columns=['day_of_week'], inplace=True)
    
    forecast = model.predict(future)
    
    # Inverse the log transformation
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    
    return forecast, model

# Build the Streamlit app layout
def main():
    st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
    
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background-color: #dfe6e9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("📈 Stock Trend Prediction Web App")
    
    st.sidebar.header("User Input Parameters")
    
    ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    st.sidebar.subheader("Prediction Parameters")
    periods = st.sidebar.slider("Days of prediction", 30, 365)
    
    if st.sidebar.button("Predict"):
        # Fetch and display stock data
        data = fetch_stock_data(ticker, start_date, end_date)
        st.subheader(f"Historical Data for {ticker}")
        st.dataframe(data.tail(), height=200)
        
        # Plot historical data
        st.subheader("Historical Stock Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title="Historical Stock Price", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig)
        
        # Predict and display trends
        forecast, model = predict_stock_trends(data, periods)
        
        st.subheader("Prediction Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), height=200)
        
        # Plotting the forecast
        st.subheader("Prediction Plot")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', name='Lower Confidence Interval'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', name='Upper Confidence Interval'))
        fig1.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Forecasted Close Price")
        st.plotly_chart(fig1)
        
        st.subheader("Prediction Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
