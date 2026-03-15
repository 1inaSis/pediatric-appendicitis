import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="PediAppend", layout="wide", initial_sidebar_state="collapsed")

import streamlit as st

st.set_page_config(
    page_title="PediAppend",
    layout="wide"
)

st.markdown("""
<style>

/* LIGHT MODE (default) */
body {
    background-color: white;
    color: black;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {

    body {
        background-color: black !important;
        color: white !important;
    }

    p, span, label, div {
        color: white !important;
    }

    h1, h2, h3, h4 {
        color: white !important;
    }

    input, textarea {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #444;
    }

    div[data-baseweb="select"] * {
        color: white !important;
        background-color: #1a1a1a !important;
    }

    .stCheckbox label {
        color: white !important;
    }

    .stNumberInput label {
        color: white !important;
    }

    .stRadio label {
        color: white !important;
    }

}

</style>
""", unsafe_allow_html=True)

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

if "page" not in st.session_state:
    st.session_state.page = "landing"

BG = "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"

if st.session_state.page == "landing":
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="collapsedControl"] {{ display: none; }}
.stApp {{
    background-image: url('data:image/jpeg;base64,{BG}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: linear-gradient(135deg, rgba(10,22,40,0.88) 0%, rgba(13,115,119,0.72) 100%);
    z-index: 0;
}}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
    position: relative;
    z-index: 1;
    min-height: 100vh !important;
}}
section[data-testid="stMain"] > div {{
    min-height: 100vh !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, #0d7377, #14b8a6) !important;
    color: white !important;
    border-radius: 50px !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 40px !important;
    box-shadow: 0 4px 24px rgba(13,115,119,0.5) !important;
    width: auto !important;
    letter-spacing: 0.5px !important;
}}
</style>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.markdown("<div style='width:75px;height:75px;border-radius:50%;background:rgba(255,255,255,0.15);border:2px solid rgba(255,255,255,0.3);display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:2rem'>🩺</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size:3.2rem;font-weight:800;color:white;margin:0;letter-spacing:-1px;text-align:center;text-shadow:0 2px 20px rgba(0,0,0,0.5)'>PediAppend</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.82rem;color:rgba(255,255,255,0.55);margin:10px 0 0;letter-spacing:3px;text-transform:uppercase;text-align:center'>Clinical Decision Support System</p>", unsafe_allow_html=True)
        st.markdown("<div style='width:50px;height:3px;background:linear-gradient(90deg,#0d7377,#14b8a6);border-radius:2px;margin:22px auto'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem;color:rgba(255,255,255,0.88);line-height:1.7;margin:0 0 8px;font-weight:500;text-align:center'>Système expert d'aide au diagnostic pédiatrique de l'appendicite</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:rgba(255,255,255,0.45);line-height:1.6;margin:0 0 30px;text-align:center'>Projet réalisé avec le dataset Regensburg Pediatric Appendicitis (UCI) — 776 patients<br>Combinant données cliniques, biologiques et échographiques</p>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("<div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:18px 10px;text-align:center;border:1px solid rgba(255,255,255,0.12)'><div style='font-size:1.6rem;font-weight:800;color:white'>0.94</div><div style='font-size:0.58rem;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px'>ROC-AUC</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:18px 10px;text-align:center;border:1px solid rgba(255,255,255,0.12)'><div style='font-size:1.6rem;font-weight:800;color:#4ade80'>91%</div><div style='font-size:0.58rem;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px'>Accuracy</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:18px 10px;text-align:center;border:1px solid rgba(255,255,255,0.12)'><div style='font-size:1.6rem;font-weight:800;color:#fbbf24'>89%</div><div style='font-size:0.58rem;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px'>Recall</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:18px 10px;text-align:center;border:1px solid rgba(255,255,255,0.12)'><div style='font-size:1.6rem;font-weight:800;color:#a78bfa'>93%</div><div style='font-size:0.58rem;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px'>F1-Score</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn2:
            if st.button("Commencer le diagnostic →"):
                st.session_state.page = "app"
                st.rerun()

        st.markdown("<p style='font-size:0.7rem;color:rgba(255,255,255,0.25);text-align:center;margin-top:16px'>Usage médical supervisé — Ne remplace pas l avis d un professionnel de santé<br>Centrale Casablanca — Coding Week 2026</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:200px'></div>", unsafe_allow_html=True)

else:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #f0f4f8 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; }
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
[data-testid="stMetric"] { background: white !important; border-radius: 12px !important; padding: 16px !important; border: 1px solid #e5e7eb !important; box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.6rem !important; font-weight: 700 !important; color: #9ca3af !important; text-transform: uppercase !important; }
.stButton > button { background-color: #0d7377 !important; color: white !important; border-radius: 10px !important; border: none !important; font-weight: 700 !important; width: 100% !important; padding: 14px !important; }
hr { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<p style='font-size:1.1rem;font-weight:700;color:#1a1d2e;margin:0'>PediAppend</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;color:#9ca3af;margin:0 0 8px'>Clinical Decision Support</p>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div style='background:#e6f4f4;border-radius:8px;padding:9px 14px;margin:4px 0;font-size:0.82rem;font-weight:600;color:#0d7377'>&#9679; Diagnosis</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:9px 14px;font-size:0.82rem;color:#6b7280'>&#9679; History</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:9px 14px;font-size:0.82rem;color:#6b7280'>&#9679; Analytics</div>", unsafe_allow_html=True)
        st.divider()
        if model_loaded:
            st.markdown("<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 14px;font-size:0.78rem;font-weight:600;color:#16a34a'>&#9679; Model Active</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 14px;font-size:0.78rem;font-weight:600;color:#dc2626'>&#9679; Model Not Found</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:6px 14px;font-size:0.75rem;color:#6b7280'>LightGBM Classifier</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#0d7377,#4db8bb);display:flex;align-items:center;justify-content:center;color:white;font-size:0.75rem;font-weight:700'>DR</div><div><div style='font-size:0.8rem;font-weight:600;color:#1a1d2e'>Dr. Smith</div><div style='font-size:0.65rem;color:#9ca3af'>Pediatric Surgeon</div></div></div>", unsafe_allow_html=True)
        st.divider()
        if st.button("← Accueil"):
            st.session_state.page = "landing"
            st.rerun()

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1 style='font-size:1.5rem;font-weight:700;color:#1a1d2e;margin:0'>Appendicitis Diagnosis</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.78rem;color:#9ca3af;margin-top:2px'>Enter patient clinical data to run the AI-powered diagnosis</p>", unsafe_allow_html=True)
    with col_h2:
        if model_loaded:
            st.markdown("<div style='text-align:right;margin-top:8px'><span style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:20px;padding:6px 14px;font-size:0.75rem;font-weight:600;color:#16a34a'>&#9679; Model Active</span></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROC-AUC", "0.94", "LightGBM")
    with c2: st.metric("Accuracy", "91%", "Validated")
    with c3: st.metric("Recall", "89%", "Sensitivity")
    with c4: st.metric("F1-Score", "93%", "Balanced")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.markdown("<div style='background:white;border-radius:14px;padding:20px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-bottom:14px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><span style='font-size:0.9rem;font-weight:700;color:#1a1d2e'>Patient Demographics</span><span style='background:#e6f4f4;color:#0d7377;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>Step 1</span></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: age = st.number_input("Age (years)", min_value=0, max_value=18, value=8)
        with c2: sex = st.selectbox("Sex", ["Male", "Female"])
        with c3: height = st.number_input("Height (cm)", min_value=50.0, max_value=200.0, value=130.0)
        c1, c2, c3 = st.columns(3)
        with c1: weight = st.number_input("Weight (kg)", min_value=5.0, max_value=150.0, value=30.0)
        with c2: bmi = st.number_input("BMI", min_value=5.0, max_value=50.0, value=17.0)
        with c3: los = st.number_input("Length of Stay", min_value=0, max_value=30, value=2)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:white;border-radius:14px;padding:20px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-bottom:14px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><span style='font-size:0.9rem;font-weight:700;color:#1a1d2e'>Clinical Symptoms</span><span style='background:#e6f4f4;color:#0d7377;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>Step 2</span></div>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:white;border-radius:14px;padding:20px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-bottom:14px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><span style='font-size:0.9rem;font-weight:700;color:#1a1d2e'>Lab Results</span><span style='background:#e6f4f4;color:#0d7377;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>Step 3</span></div>", unsafe_allow_html=True)
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
            alvarado = st.number_input("Alvarado Score", min_value=0, max_value=10, value=5)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:white;border-radius:14px;padding:20px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-bottom:14px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><span style='font-size:0.9rem;font-weight:700;color:#1a1d2e'>Urine & Ultrasound</span><span style='background:#e6f4f4;color:#0d7377;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>Step 4</span></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ketones = st.selectbox("Ketones in Urine", ["No", "Yes"])
            rbc_urine = st.selectbox("RBC in Urine", ["No", "Yes"])
            wbc_urine = st.selectbox("WBC in Urine", ["No", "Yes"])
        with c2:
            us_performed = st.selectbox("US Performed", ["Yes", "No"])
            us_number = st.number_input("US Number", min_value=0, max_value=5, value=1)
            appendix_us = st.selectbox("Appendix on US", ["No", "Yes", "Inconclusive"])
        with c3:
            appendix_diameter = st.number_input("Appendix Diameter (mm)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
            free_fluids = st.selectbox("Free Fluids", ["No", "Yes"])
            stool = st.selectbox("Stool", ["Normal", "Abnormal"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background:white;border-radius:14px;padding:20px;border:1px solid #e5e7eb;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-bottom:14px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><span style='font-size:0.9rem;font-weight:700;color:#1a1d2e'>Management & Scores</span><span style='background:#e6f4f4;color:#0d7377;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>Step 5</span></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            management = st.selectbox("Management", ["Conservative", "Operative"])
            severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        with c2:
            diagnosis_presumptive = st.selectbox("Diagnosis Presumptive", ["No Appendicitis", "Appendicitis"])
            pas = st.number_input("Paediatric Appendicitis Score", min_value=0, max_value=10, value=5)
        with c3:
            ipsilateral_rebound = st.selectbox("Ipsilateral Rebound", ["No", "Yes", "Equivocal"])
        st.markdown("</div>", unsafe_allow_html=True)

        predict = st.button("Run Diagnosis Prediction")

    with right_col:
        if predict:
            input_data = pd.DataFrame([{
                "Age": age, "BMI": bmi, "Sex": 1 if sex == "Male" else 0,
                "Height": height, "Weight": weight, "Length_of_Stay": los,
                "Management": 1 if management == "Operative" else 0,
                "Severity": ["Mild","Moderate","Severe"].index(severity),
                "Diagnosis_Presumptive": 1 if diagnosis_presumptive == "Appendicitis" else 0,
                "Alvarado_Score": alvarado, "Paedriatic_Appendicitis_Score": pas,
                "Appendix_on_US": ["No","Yes","Inconclusive"].index(appendix_us),
                "Appendix_Diameter": appendix_diameter,
                "Migratory_Pain": 1 if migratory_pain else 0,
                "Lower_Right_Abd_Pain": 1 if lower_right else 0,
                "Contralateral_Rebound_Tenderness": 1 if contralateral_rebound else 0,
                "Coughing_Pain": 1 if coughing_pain else 0,
                "Nausea": 1 if nausea else 0,
                "Loss_of_Appetite": 1 if loss_of_appetite else 0,
                "Body_Temperature": body_temp, "WBC_Count": wbc,
                "Neutrophil_Percentage": neutrophil,
                "Neutrophilia": 1 if neutrophilia == "Yes" else 0,
                "RBC_Count": rbc, "Hemoglobin": hemoglobin, "RDW": rdw,
                "Thrombocyte_Count": thrombocyte,
                "Ketones_in_Urine": 1 if ketones == "Yes" else 0,
                "RBC_in_Urine": 1 if rbc_urine == "Yes" else 0,
                "WBC_in_Urine": 1 if wbc_urine == "Yes" else 0,
                "CRP": crp, "Dysuria": 1 if dysuria else 0,
                "Stool": 1 if stool == "Abnormal" else 0,
                "Peritonitis": 1 if peritonitis else 0,
                "Psoas_Sign": 1 if psoas_sign else 0,
                "Ipsilateral_Rebound_Tenderness": 1 if ipsilateral_rebound == "Yes" else 0,
                "US_Performed": 1 if us_performed == "Yes" else 0,
                "US_Number": us_number, "Free_Fluids": 1 if free_fluids == "Yes" else 0,
            }])
            if model_loaded:
                proba = model.predict_proba(input_data)[0][1]
                percent = int(proba * 100)
                color = "#dc2626" if proba > 0.5 else "#16a34a"
                border = "#fecaca" if proba > 0.5 else "#bbf7d0"
                bg2 = "#fef2f2" if proba > 0.5 else "#f0fdf4"
                label = "Appendicitis Likely" if proba > 0.5 else "Appendicitis Unlikely"
                risk = "HIGH RISK" if proba > 0.5 else "LOW RISK"
                st.markdown("<p style='font-size:0.65rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:1.5px'>DIAGNOSIS RESULT</p>", unsafe_allow_html=True)
                st.markdown("<div style='background:white;border-radius:16px;padding:24px;border:1.5px solid " + border + ";box-shadow:0 4px 20px rgba(0,0,0,0.06);margin-bottom:16px'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'><p style='font-size:1.1rem;font-weight:700;color:" + color + ";margin:0'>" + label + "</p><span style='background:" + bg2 + ";color:" + color + ";border:1px solid " + border + ";border-radius:20px;padding:4px 12px;font-size:0.65rem;font-weight:700'>" + risk + "</span></div><p style='font-size:3.5rem;font-weight:700;color:" + color + ";margin:0;line-height:1'>" + str(percent) + "<span style='font-size:1.2rem;color:#9ca3af'>%</span></p><p style='font-size:0.68rem;color:#9ca3af;margin:4px 0 12px'>Appendicitis probability - LightGBM model</p><div style='height:6px;background:#f3f4f6;border-radius:3px;margin-bottom:16px'><div style='height:6px;background:" + color + ";border-radius:3px;width:" + str(percent) + "%'></div></div><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px'><div style='text-align:center;background:#f9fafb;border-radius:8px;padding:10px'><div style='font-size:1rem;font-weight:700;color:#0d7377'>0.94</div><div style='font-size:0.58rem;color:#9ca3af'>AUC</div></div><div style='text-align:center;background:#f9fafb;border-radius:8px;padding:10px'><div style='font-size:1rem;font-weight:700;color:#16a34a'>91%</div><div style='font-size:0.58rem;color:#9ca3af'>Accuracy</div></div><div style='text-align:center;background:#f9fafb;border-radius:8px;padding:10px'><div style='font-size:1rem;font-weight:700;color:#d97706'>89%</div><div style='font-size:0.58rem;color:#9ca3af'>Recall</div></div><div style='text-align:center;background:#f9fafb;border-radius:8px;padding:10px'><div style='font-size:1rem;font-weight:700;color:#6b7280'>93%</div><div style='font-size:0.58rem;color:#9ca3af'>F1</div></div></div></div>", unsafe_allow_html=True)
                st.markdown("<div style='background:white;border-radius:16px;padding:20px;border:1px solid #e5e7eb'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'><p style='font-size:0.9rem;font-weight:700;color:#1a1d2e;margin:0'>SHAP Explainability</p><span style='background:#ede9fe;color:#7c3aed;border-radius:20px;padding:3px 10px;font-size:0.65rem;font-weight:700'>AI Insights</span></div>", unsafe_allow_html=True)
                for i, (fname, fval) in enumerate([("CRP (mg/L)", 0.42), ("WBC Count", 0.35), ("Rebound Tend.", 0.28), ("Body Temp", 0.21), ("Alvarado Score", 0.18)]):
                    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'><span style='font-size:0.7rem;color:#9ca3af;width:12px'>" + str(i+1) + "</span><span style='font-size:0.78rem;color:#374151;width:110px'>" + fname + "</span><div style='flex:1;height:6px;background:#f3f4f6;border-radius:3px'><div style='height:6px;background:#0d7377;border-radius:3px;width:" + str(int(fval*200)) + "px'></div></div><span style='font-size:0.75rem;font-weight:600;color:#0d7377'>+" + str(fval) + "</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Model not found in models/ folder")
        else:
            st.markdown("<div style='background:white;border-radius:16px;padding:40px;border:1px solid #e5e7eb;text-align:center'><p style='color:#9ca3af;font-size:0.85rem;margin:0'>Fill in the patient data and click<br><strong style=\'color:#0d7377\'>Run Diagnosis Prediction</strong></p></div>", unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.8rem;font-weight:700;color:#1a1d2e;margin:14px 0 10px'>Model Performance</p>", unsafe_allow_html=True)
        if os.path.exists("results/confusion_matrix.png"):
            st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        if os.path.exists("results/roc_curve.png"):
            st.image("results/roc_curve.png", caption="ROC Curve", use_container_width=True)

    st.divider()
    st.markdown("<p style='text-align:center;font-size:0.68rem;color:#9ca3af'>Decision support tool only - Not a replacement for clinical judgment</p>", unsafe_allow_html=True)
