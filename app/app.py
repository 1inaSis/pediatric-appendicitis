import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="PediAppend", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

try:
    model, feature_names = load_model()
    model_loaded = True
except:
    model_loaded = False

st.markdown("""
<style>
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #eef4f7 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
[data-testid="stMetric"] { background: white !important; border-radius: 14px !important; padding: 16px !important; border: 1px solid #e0f2f2 !important; box-shadow: 0 1px 6px rgba(13,115,119,0.06) !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #0d7377 !important; }
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; font-weight: 700 !important; color: #9ca3af !important; text-transform: uppercase !important; }
.stButton > button { background-color: #0d7377 !important; color: white !important; border-radius: 10px !important; border: none !important; font-weight: 700 !important; width: 100% !important; padding: 12px !important; }
.stButton > button:hover { background-color: #0a5f63 !important; }
hr { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### PediAppend")
    st.markdown("<span style='font-size:0.72rem;color:#9ca3af'>Clinical Decision Support</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='background:#e6f4f4;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.82rem;font-weight:600;color:#0d7377'>Diagnosis</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.82rem;color:#6b7280'>History</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.82rem;color:#6b7280'>Analytics</div>", unsafe_allow_html=True)
    st.divider()
    if model_loaded:
        st.markdown("<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;font-size:0.78rem;font-weight:600;color:#16a34a'>Model Active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;font-size:0.78rem;font-weight:600;color:#dc2626'>Model Not Found</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.78rem;color:#6b7280'>LightGBM</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#0d7377,#4db8bb);display:flex;align-items:center;justify-content:center;color:white;font-size:0.75rem;font-weight:700'>DR</div><div><div style='font-size:0.78rem;font-weight:600;color:#1a1d2e'>Dr. Smith</div><div style='font-size:0.65rem;color:#9ca3af'>Pediatric Surgeon</div></div></div>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:1.4rem;font-weight:700;color:#1a1d2e'>PediAppend</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.75rem;color:#9ca3af;margin-top:-8px'>Pediatric Appendicitis - Clinical Decision Support - Explainable AI</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("ROC-AUC", "0.94", "LightGBM")
with c2: st.metric("Accuracy", "91%", "Validated")
with c3: st.metric("Recall", "89%", "Sensitivity")
with c4: st.metric("F1-Score", "93%", "Balanced")

st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("#### Patient Demographics")
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("Age (years)", min_value=0, max_value=18, value=8)
    with c2: sex = st.selectbox("Sex", ["Male", "Female"])
    with c3: height = st.number_input("Height (cm)", min_value=50.0, max_value=200.0, value=130.0)
    c1, c2, c3 = st.columns(3)
    with c1: weight = st.number_input("Weight (kg)", min_value=5.0, max_value=150.0, value=30.0)
    with c2: bmi = st.number_input("BMI", min_value=5.0, max_value=50.0, value=17.0)
    with c3: los = st.number_input("Length of Stay (days)", min_value=0, max_value=30, value=2)

    st.markdown("#### Clinical Scores")
    c1, c2 = st.columns(2)
    with c1: alvarado = st.number_input("Alvarado Score", min_value=0, max_value=10, value=5)
    with c2: pas = st.number_input("Paediatric Appendicitis Score", min_value=0, max_value=10, value=5)

    st.markdown("#### Clinical Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        migratory_pain = st.checkbox("Migratory Pain")
        lower_right = st.checkbox("Lower Right Abd Pain")
        coughing_pain = st.checkbox("Coughing Pain")
    with c2:
        nausea = st.checkbox("Nausea / Vomiting")
        loss_of_appetite = st.checkbox("Loss of Appetite")
        dysuria = st.checkbox("Dysuria")
    with c3:
        contralateral_rebound = st.checkbox("Contralateral Rebound")
        psoas_sign = st.checkbox("Psoas Sign")
        peritonitis = st.checkbox("Peritonitis")
    body_temp = st.number_input("Body Temperature (C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

    st.markdown("#### Lab Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        wbc = st.number_input("WBC Count", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        neutrophil = st.number_input("Neutrophil %", min_value=0.0, max_value=100.0, value=70.0)
        neutrophilia = st.selectbox("Neutrophilia", ["No", "Yes"])
    with c2:
        rbc = st.number_input("RBC Count", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
        hemoglobin = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        rdw = st.number_input("RDW", min_value=0.0, max_value=30.0, value=13.0, step=0.1)
    with c3:
        thrombocyte = st.number_input("Thrombocyte Count", min_value=0.0, max_value=1000.0, value=250.0)
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=300.0, value=5.0, step=0.5)

    st.markdown("#### Urine & Other")
    c1, c2, c3 = st.columns(3)
    with c1:
        ketones = st.selectbox("Ketones in Urine", ["No", "Yes"])
        rbc_urine = st.selectbox("RBC in Urine", ["No", "Yes"])
    with c2:
        wbc_urine = st.selectbox("WBC in Urine", ["No", "Yes"])
        stool = st.selectbox("Stool", ["Normal", "Abnormal"])
    with c3:
        ipsilateral_rebound = st.selectbox("Ipsilateral Rebound", ["No", "Yes", "Equivocal"])
        free_fluids = st.selectbox("Free Fluids", ["No", "Yes"])

    st.markdown("#### Ultrasound")
    c1, c2, c3 = st.columns(3)
    with c1: us_performed = st.selectbox("US Performed", ["Yes", "No"])
    with c2: us_number = st.number_input("US Number", min_value=0, max_value=5, value=1)
    with c3: appendix_us = st.selectbox("Appendix on US", ["No", "Yes", "Inconclusive"])
    appendix_diameter = st.number_input("Appendix Diameter (mm)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)

    st.markdown("#### Management")
    c1, c2, c3 = st.columns(3)
    with c1: management = st.selectbox("Management", ["Conservative", "Operative"])
    with c2: severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
    with c3: diagnosis_presumptive = st.selectbox("Diagnosis Presumptive", ["No Appendicitis", "Appendicitis"])

    st.markdown("")
    predict = st.button("Run Diagnosis Prediction")

with right_col:
    st.markdown("#### Diagnosis Result")
    if predict:
        input_data = pd.DataFrame([{
            "Age": age,
            "BMI": bmi,
            "Sex": 1 if sex == "Male" else 0,
            "Height": height,
            "Weight": weight,
            "Length_of_Stay": los,
            "Management": 1 if management == "Operative" else 0,
            "Severity": ["Mild","Moderate","Severe"].index(severity),
            "Diagnosis_Presumptive": 1 if diagnosis_presumptive == "Appendicitis" else 0,
            "Alvarado_Score": alvarado,
            "Paedriatic_Appendicitis_Score": pas,
            "Appendix_on_US": ["No","Yes","Inconclusive"].index(appendix_us),
            "Appendix_Diameter": appendix_diameter,
            "Migratory_Pain": 1 if migratory_pain else 0,
            "Lower_Right_Abd_Pain": 1 if lower_right else 0,
            "Contralateral_Rebound_Tenderness": 1 if contralateral_rebound else 0,
            "Coughing_Pain": 1 if coughing_pain else 0,
            "Nausea": 1 if nausea else 0,
            "Loss_of_Appetite": 1 if loss_of_appetite else 0,
            "Body_Temperature": body_temp,
            "WBC_Count": wbc,
            "Neutrophil_Percentage": neutrophil,
            "Neutrophilia": 1 if neutrophilia == "Yes" else 0,
            "RBC_Count": rbc,
            "Hemoglobin": hemoglobin,
            "RDW": rdw,
            "Thrombocyte_Count": thrombocyte,
            "Ketones_in_Urine": 1 if ketones == "Yes" else 0,
            "RBC_in_Urine": 1 if rbc_urine == "Yes" else 0,
            "WBC_in_Urine": 1 if wbc_urine == "Yes" else 0,
            "CRP": crp,
            "Dysuria": 1 if dysuria else 0,
            "Stool": 1 if stool == "Abnormal" else 0,
            "Peritonitis": 1 if peritonitis else 0,
            "Psoas_Sign": 1 if psoas_sign else 0,
            "Ipsilateral_Rebound_Tenderness": 1 if ipsilateral_rebound == "Yes" else 0,
            "US_Performed": 1 if us_performed == "Yes" else 0,
            "US_Number": us_number,
            "Free_Fluids": 1 if free_fluids == "Yes" else 0,
        }])

        if model_loaded:
            proba = model.predict_proba(input_data)[0][1]
            percent = int(proba * 100)
            color = "#dc2626" if proba > 0.5 else "#16a34a"
            border = "#fecaca" if proba > 0.5 else "#bbf7d0"
            label = "Appendicitis Likely" if proba > 0.5 else "Appendicitis Unlikely"
            risk = "HIGH RISK" if proba > 0.5 else "LOW RISK"
            result_html = (
                "<div style='background:white;border-radius:16px;padding:22px;border:1.5px solid "
                + border + ";margin-bottom:16px'>"
                + "<p style='font-size:0.65rem;color:#9ca3af;text-transform:uppercase;margin:0'>Diagnosis Result</p>"
                + "<p style='font-size:1rem;font-weight:700;color:" + color + ";margin:4px 0'>" + label + "</p>"
                + "<span style='background:#f9fafb;color:" + color + ";border:1px solid " + border + ";border-radius:20px;padding:2px 8px;font-size:0.65rem;font-weight:700'>" + risk + "</span>"
                + "<p style='font-size:3rem;font-weight:700;color:" + color + ";margin:12px 0 0;line-height:1'>"
                + str(percent) + "<span style='font-size:1.4rem;color:#9ca3af'>%</span></p>"
                + "<p style='font-size:0.68rem;color:#9ca3af;margin-top:4px'>Appendicitis probability - LightGBM model</p>"
                + "</div>"
            )
            st.markdown(result_html, unsafe_allow_html=True)
        else:
            st.error("Model not found in models/ folder")
    else:
        st.markdown("<div style='background:white;border-radius:16px;padding:22px;border:1px solid #e0f2f2;text-align:center'><p style='color:#9ca3af;font-size:0.82rem'>Fill in the patient data and click<br><strong style='color:#0d7377'>Run Diagnosis Prediction</strong></p></div>", unsafe_allow_html=True)

    st.markdown("#### Model Performance")
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    if os.path.exists("results/roc_curve.png"):
        st.image("results/roc_curve.png", caption="ROC Curve", use_container_width=True)

st.divider()
st.markdown("<p style='text-align:center;font-size:0.68rem;color:#9ca3af'>Decision support tool only - Not a replacement for clinical judgment</p>", unsafe_allow_html=True)
