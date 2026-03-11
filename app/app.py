import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PediAppend", layout="wide")

st.title("PediAppend")
st.caption("Pediatric Appendicitis Clinical Decision Support")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.number_input("Age (years)", min_value=0, max_value=18, value=8)
    sex = st.selectbox("Sex", ["Male", "Female"])
    st.subheader("Clinical Symptoms")
    migratory_pain = st.checkbox("Migratory Pain (pain moves to lower right)")
    nausea = st.checkbox("Nausea / Vomiting")
    loss_of_appetite = st.checkbox("Loss of Appetite")
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

with col2:
    st.subheader("Physical Examination")
    contralateral_rebound = st.checkbox("Contralateral Rebound Tenderness")
    ipsilateral_rebound = st.selectbox("Ipsilateral Rebound Tenderness", ["No", "Yes", "Equivocal"])
    st.subheader("Lab Results")
    wbc = st.number_input("WBC Count (×10³/µL)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=300.0, value=5.0, step=0.5)

st.divider()
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict = st.button("Run Diagnosis Prediction")

if predict:
    input_data = pd.DataFrame([{
        "Age": age, "Sex": 1 if sex == "Male" else 0,
        "Migratory_Pain": 1 if migratory_pain else 0,
        "Nausea": 1 if nausea else 0,
        "Loss_of_Appetite": 1 if loss_of_appetite else 0,
        "Body_Temperature": body_temp,
        "Contralateral_Rebound_Tenderness": 1 if contralateral_rebound else 0,
        "Ipsilateral_Rebound_Tenderness": 1 if ipsilateral_rebound == "Yes" else 0,
        "WBC_Count": wbc, "CRP": crp
    }])
    st.success("Data collected successfully!")
    st.dataframe(input_data)
    st.warning("Model not yet connected - waiting for P3 model file")

st.caption("Decision support tool only - not a replacement for clinical judgment.")