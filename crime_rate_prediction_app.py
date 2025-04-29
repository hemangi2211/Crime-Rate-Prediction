import streamlit as st
import joblib
import numpy as np

# Set title
st.title(" Crime Rate Prediction App")
st.write("Enter the required features to predict the violent crime rate.")

# Load model and scaler
model = joblib.load("crime_rate_model.pkl")
scaler = joblib.load("crime_scaler.pkl")

# Feature input fields — replace/add as per your dataset features
f1 = st.number_input("Population", min_value=0)
f2 = st.number_input("Murder and Nonnegligent Manslaughter", min_value=0)
f3 = st.number_input("Robbery", min_value=0)
f4 = st.number_input("Aggravated Assault", min_value=0)
f5 = st.number_input("Property Crime", min_value=0)
f6 = st.number_input("Burglary", min_value=0)
f7 = st.number_input("Larceny-theft", min_value=0)
f8 = st.number_input("Motor Vehicle Theft", min_value=0)

# Collect input into a list
features = [[f1, f2, f3, f4, f5, f6, f7, f8]]

# Predict button
if st.button("Predict Crime Rate"):
    try:
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        st.success(f"Predicted Violent Crime Rate: {prediction[0]:.2f} per 100,000 people")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
