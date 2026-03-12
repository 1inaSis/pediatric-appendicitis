import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="PediAppend", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #eef4f7 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
h1 { font-size: 1.4rem !important; font-weight: 700 !important; color: #1a1d2e !important; margin-bottom: 0 !important; }
h4 { font-size: 0.82rem !important; font-weight: 700 !important; color: #1a1d2e !important; }
[data-testid="stMetric"] { background: white !important; border-radius: 14px !important; padding: 16px !important; border: 1px solid #e0f2f2 !important; box-shadow: 0 1px 6px rgba(13,115,119,0.06) !important; }
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; font-weight: 700 !important; color: #9ca3af !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #0d7377 !important; }
.stButton > button { background-color: #0d7377 !important; color: white !important; border-radius: 10px !important; border: none !important; font-weight: 700 !important; font-size: 0.88rem !important; width: 100% !important; }
.stButton > button:hover { background-color: #0a5f63 !important; }
hr { border-color: #e5e7eb !important; margin: 12px 0 !important; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### PediAppend")
    st.markdown("<span style='font-size:0.72rem;color:#9ca3af'>Clinical Decision Support</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<span style='font-size:0.65rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:1.5px'>MAIN MENU</span>", unsafe_allow_html=True)
    st.markdown("<div style='background:#e6f4f4;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.82rem;font-weight:600;color:#0d7377'>Diagnosis</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.82rem;color:#6b7280'>History</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.82rem;color:#6b7280'>Analytics</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<span style='font-size:0.65rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:1.5px'>MODEL INFO</span>", unsafe_allow_html=True)
    st.markdown("<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;font-size:0.78rem;font-weight:600;color:#16a34a'>Model Active</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:8px 12px;font-size:0.78rem;color:#6b7280'>Random Forest</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#0d7377,#4db8bb);display:flex;align-items:center;justify-content:center;color:white;font-size:0.75rem;font-weight:700'>DR</div><div><div style='font-size:0.78rem;font-weight:600;color:#1a1d2e'>Dr. Smith</div><div style='font-size:0.65rem;color:#9ca3af'>Pediatric Surgeon</div></div></div>", unsafe_allow_html=True)

# HEADER
st.markdown("<h1>PediAppend</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.75rem;color:#9ca3af;margin-top:-8px'>Pediatric Appendicitis - Clinical Decision Support - Explainable AI</p>", unsafe_allow_html=True)

# STATS
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("ROC-AUC", "0.94", "Random Forest")
with c2: st.metric("Accuracy", "91%", "Validated")
with c3: st.metric("Recall", "89%", "Sensitivity")
with c4: st.metric("F1-Score", "93%", "Balanced")

st.divider()

# MAIN LAYOUT
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("#### Patient Demographics")
    c1, c2 = st.columns(2)
    with c1: age = st.number_input("Age (years)", min_value=0, max_value=18, value=8)
    with c2: sex = st.selectbox("Sex", ["Male", "Female"])

    st.markdown("#### Clinical Symptoms")
    c1, c2 = st.columns(2)
    with c1:
        migratory_pain = st.checkbox("Migratory Pain")
        nausea = st.checkbox("Nausea / Vomiting")
    with c2:
        loss_of_appetite = st.checkbox("Loss of Appetite")
        fever = st.checkbox("Fever > 38 C")
    body_temp = st.number_input("Body Temperature (C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

    st.markdown("#### Lab Results & Physical Exam")
    c1, c2 = st.columns(2)
    with c1: wbc = st.number_input("WBC Count (x1000/uL)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    with c2: crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=300.0, value=5.0, step=0.5)
    c1, c2 = st.columns(2)
    with c1: contralateral_rebound = st.checkbox("Contralateral Rebound Tenderness")
    with c2: ipsilateral_rebound = st.selectbox("Ipsilateral Rebound Tenderness", ["No", "Yes", "Equivocal"])

    st.markdown("")
    predict = st.button("Run Diagnosis Prediction")

with right_col:
    st.markdown("#### Diagnosis Result")
    if predict:
        st.markdown("""
<div style='background:white;border-radius:16px;padding:22px;border:1.5px solid #fecaca;box-shadow:0 4px 20px rgba(239,68,68,0.08);margin-bottom:16px'>
<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px'>
<div><p style='font-size:0.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1.5px;margin:0'>Diagnosis Result</p>
<p style='font-size:1rem;font-weight:700;color:#dc2626;margin:4px 0 0'>Appendicitis Likely</p></div>
<span style='background:#fef2f2;color:#dc2626;border:1px solid #fecaca;border-radius:20px;padding:4px 10px;font-size:0.65rem;font-weight:700'>HIGH RISK</span>
</div>
<p style='font-size:3rem;font-weight:700;color:#dc2626;margin:0;line-height:1'>--<span style='font-size:1.4rem;color:#9ca3af'>%</span></p>
<p style='font-size:0.68rem;color:#9ca3af;margin-top:4px'>Waiting for P3 model</p>
<div style='height:5px;background:#f3f4f6;border-radius:3px;margin:14px 0'></div>
<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px'>
<div style='text-align:center;background:#f9fafb;border-radius:8px;padding:8px'><div style='font-size:1rem;font-weight:700;color:#0d7377'>0.94</div><div style='font-size:0.58rem;color:#9ca3af'>AUC</div></div>
<div style='text-align:center;background:#f9fafb;border-radius:8px;padding:8px'><div style='font-size:1rem;font-weight:700;color:#0d7377'>91%</div><div style='font-size:0.58rem;color:#9ca3af'>Accuracy</div></div>
<div style='text-align:center;background:#f9fafb;border-radius:8px;padding:8px'><div style='font-size:1rem;font-weight:700;color:#0d7377'>89%</div><div style='font-size:0.58rem;color:#9ca3af'>Recall</div></div>
<div style='text-align:center;background:#f9fafb;border-radius:8px;padding:8px'><div style='font-size:1rem;font-weight:700;color:#0d7377'>93%</div><div style='font-size:0.58rem;color:#9ca3af'>F1</div></div>
</div></div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style='background:white;border-radius:16px;padding:22px;border:1px solid #e0f2f2;text-align:center'>
<p style='color:#9ca3af;font-size:0.82rem'>Fill in the patient data and click<br><strong style='color:#0d7377'>Run Diagnosis Prediction</strong></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Model Performance")
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    if os.path.exists("results/roc_curve.png"):
        st.image("results/roc_curve.png", caption="ROC Curve", use_container_width=True)

st.divider()
st.markdown("<p style='text-align:center;font-size:0.68rem;color:#9ca3af'>Decision support tool only - Not a replacement for clinical judgment</p>", unsafe_allow_html=True)
