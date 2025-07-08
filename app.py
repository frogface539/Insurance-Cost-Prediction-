import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="💰 Insurance Cost Predictor", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2e7d32;'>💰 Insurance Cost Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your estimated medical insurance charges using a trained <b>Artificial Neural Network (ANN)</b> model.</p>", unsafe_allow_html=True)
st.markdown("---")

model = load_model("model/insurance_ann_model.h5")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.markdown("### 🧾 Customer Information")

left_col, right_col = st.columns(2)

with left_col:
    age = st.slider("Age", 18, 100, 35)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 28.0, step=0.1)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

with right_col:
    sex = st.radio("Gender", ["Male", "Female"], horizontal=True)
    smoker = st.radio("Smoker?", ["Yes", "No"], horizontal=True)

st.markdown("---")

center_col = st.columns(3)[1]

with center_col:
    if st.button("🔍 Predict Insurance Charges"):
        sex_encoded = 1 if sex == "Male" else 0
        smoker_encoded = 1 if smoker == "Yes" else 0
        region_northwest = 1 if region == "northwest" else 0
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex_encoded,
            "bmi": bmi,
            "children": children,
            "smoker": smoker_encoded,
            "region_northwest": region_northwest,
            "region_southeast": region_southeast,
            "region_southwest": region_southwest
        }])

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0][0]
        prediction = max(0, prediction)

        st.markdown("### 🔎 Prediction Result")
        st.success(f"💸 Estimated Insurance Charges: ₹{prediction:,.2f}")
