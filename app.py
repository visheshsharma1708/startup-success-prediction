import streamlit as st
import pickle
import numpy as np

st.title("Startup Profit Prediction")

rd = st.number_input("R&D Spend", 0.0)
admin = st.number_input("Administration Spend", 0.0)
marketing = st.number_input("Marketing Spend", 0.0)
state = st.selectbox("State", ['California', 'Florida', 'New York'])

# Map state to one-hot encoding
state_florida = 1 if state == "Florida" else 0
state_newyork = 1 if state == "New York" else 0

input_data = np.array([[rd, admin, marketing, state_florida, state_newyork]])

# Load model
model = pickle.load(open("startup_profit_model.pkl", "rb"))

if st.button("Predict Profit"):
    result = model.predict(input_data)[0]
    st.success(f"Predicted Profit: ${round(result, 2)}")
