import streamlit as st

st.set_page_config(page_title="PediAppend", layout="wide")

st.title("PediAppend")
st.caption("Pediatric Appendicitis Clinical Decision Support")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.number_input("Age (years)", min_value=0, max_value=18, value=8)
    sex = st.selectbox("Sex", ["Male", "Female"])

    st.subheader("Symptoms")
    migratory_pain = st.checkbox("Migratory Pain")
    nausea = st.checkbox("Nausea / Vomiting")
    loss_of_appetite = st.checkbox("Loss of Appetite")
    fever = st.checkbox("Fever > 38C")

with col2:
    st.subheader("Physical Examination")
    rebound_tenderness = st.checkbox("Rebound Tenderness")
    guarding = st.checkbox("Guarding / Rigidity")
    temperature = st.number_input("Temperature (C)", min_value=35.0, max_value=42.0, value=37.0)

    st.subheader("Lab Results")
    wbc = st.number_input("WBC x1000/uL", min_value=0.0, max_value=50.0, value=10.0)
    crp = st.number_input("CRP mg/L", min_value=0.0, max_value=300.0, value=5.0)

st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict = st.button("Run Diagnosis Prediction")

if predict:
    st.warning("Model not yet connected - coming Day 3!")

st.caption("Decision support tool only.")