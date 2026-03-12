# PediAppend — Pediatric Appendicitis Clinical Decision Support

An explainable AI-powered web application for supporting the diagnosis of appendicitis in pediatric patients.

## Project Overview

PediAppend uses machine learning to predict the probability of appendicitis in children based on clinical, biological, and demographic features. The app provides real-time predictions with model performance metrics.

## Features

- AI-powered diagnosis prediction (Random Forest)
- 10 clinical input features
- Model performance visualization (ROC Curve, Confusion Matrix)
- Clean medical-grade UI built with Streamlit
- Dockerized for easy deployment
- CI/CD pipeline with GitHub Actions

## Input Features

| Feature | Description |
|--------|-------------|
| Age | Patient age in years |
| Sex | Male / Female |
| Migratory Pain | Pain moving to lower right |
| Nausea | Nausea or vomiting |
| Loss of Appetite | Anorexia |
| Body Temperature | In Celsius |
| Contralateral Rebound | Rebound tenderness |
| Ipsilateral Rebound | Ipsilateral tenderness |
| WBC Count | White blood cell count |
| CRP | C-reactive protein |

## Installation
```bash
git clone https://github.com/1inaSis/pediatric-appendicitis.git
cd pediatric-appendicitis
pip install -r requirements.txt
streamlit run app/app.py
```

## Docker
```bash
docker build -t pediappend .
docker run -p 8501:8501 pediappend
```

## Dataset

Regensburg Pediatric Appendicitis Dataset — UCI Machine Learning Repository

## Team

Centrale Casablanca — Coding Week March 2026

