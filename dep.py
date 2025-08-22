import streamlit as st
import joblib
import numpy as np


model = joblib.load("waterpotability_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" Water Potability Predictor")
st.write("Water Potability Data Analysis involves examining water quality parameters to determine whether water is safe for human consumption.")


ph = st.number_input("Enter your pH")
Hardness = st.number_input("Enter your Hardness")
Solids = st.number_input("Enter Solids")
Chloramines = st.number_input("Enter Chloramines")
Sulfate = st.number_input("Enter Sulfate")
Conductivity = st.number_input("Enter Conductivity")
Organic_carbon = st.number_input("Enter Organic Carbon")
Trihalomethanes = st.number_input("Enter Trihalomethanes")
Turbidity = st.number_input("Enter Turbidity")


if st.button("Predict"):
  
    user_input = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

    
    user_input_scaled = scaler.transform(user_input)

    
    prediction = model.predict(user_input_scaled)
    proba = model.predict_proba(user_input_scaled)

  
    if prediction[0] == 1:
        st.success(" Prediction: POTABLE (1)")
    else:
        st.error("Prediction: NOT POTABLE (0)")

   



