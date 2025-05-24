
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model #will be used later
import matplotlib.pyplot as plt
import yfinance as yf
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("📈 Stock Price Predictor App")

# Apply custom background
import base64

def set_background(images):
    with open(images, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .stApp {
            color: white;  
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images.jpg")






############################################################################################################
############################################################################################################

stock = st.text_input("Enter the Stock ID", "IRFC.NS")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

st.subheader("📊 Stock Data")
st.write(google_data)

model_path = "Google_stock_price_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"🚨 Model file '{model_path}' not found!")
    st.stop()

if 'Close' not in google_data.columns:
    st.error("🚨 Column 'Close' not found in data!")
    st.stop()

splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.iloc[splitting_len:]['Close'])

st.subheader('📉 Original Close Price and Moving Averages')

for days in [250, 200, 100]:
    google_data[f'MA_for_{days}_days'] = google_data['Close'].rolling(days).mean()
    fig = plt.figure(figsize=(15, 6))
    plt.plot(google_data['Close'], label="Original Close Price", color='b')
    plt.plot(google_data[f'MA_for_{days}_days'], label=f'MA for {days} days', color='orange')
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

scaler = MinMaxScaler(feature_range=(0, 1))
if not x_test.empty:
    scaled_data = scaler.fit_transform(x_test)
else:
    st.error("🚨 No test data available after split!")
    st.stop()

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.reshape(-1),
    'Predictions': inv_predictions.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

st.subheader("📊 Original vs Predicted Stock Prices")
st.write(plotting_data)

st.subheader('📉 Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data['Close'][:splitting_len + 100], label="Data - Not Used", color="gray")
plt.plot(plotting_data['Original Test Data'], label="Original Test Data", color="blue")
plt.plot(plotting_data['Predictions'], label="Predicted Data", color="red")
plt.legend()
st.pyplot(fig)
plt.close(fig)



