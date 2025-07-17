import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Bitcoin Price Predictor", layout="centered")

st.title("📈 Bitcoin Price Prediction App")
st.markdown("Predict future Bitcoin prices using LSTM model and blockchain data.")

# Load model and scaler
model = load_model("model/bitcoin_lstm_model.h5")
with open("scaler/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Show a preview of previous 60 days (simulated input)
st.subheader("⚙️ Provide Last 60 Days Close Prices")

input_data = st.text_area("Paste 60 closing prices (comma-separated):", 
                          "20000,20100,20200,...", height=150)

if st.button("Predict"):
    try:
        # Convert text to array
        input_list = list(map(float, input_data.strip().split(",")))
        if len(input_list) != 60:
            st.error("❌ Please provide exactly 60 numbers.")
        else:
            scaled_input = scaler.transform(np.array(input_list).reshape(-1, 1))
            sequence = scaled_input.reshape(1, 60, 1)

            predictions = []
            for _ in range(7):
                pred_scaled = model.predict(sequence)[0][0]
                predictions.append(pred_scaled)
                sequence = np.append(sequence[:, 1:, :], [[[pred_scaled]]], axis=1)

            future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            st.success("✅ Prediction Complete")
            st.subheader("📅 Predicted Prices for Next 7 Days")
            for i, price in enumerate(future_prices):
                st.write(f"Day {i+1}: *${price:.2f}*")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")