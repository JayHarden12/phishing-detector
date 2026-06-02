import pandas as pd
import numpy as np
import joblib
import sqlite3
import datetime
import streamlit as st
import os

# Database initialization
def init_db():
    conn = sqlite3.connect('phishing_detector.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (timestamp TEXT, url_id TEXT, model TEXT, predicted_label TEXT, 
                  confidence REAL, rejection_status TEXT)''')
    conn.commit()
    conn.close()

# Load models and data safely
@st.cache_resource
def load_resources():
    models = {}
    try:
        models['Naive Bayes'] = joblib.load('models/naive_bayes.joblib')
        models['Logistic Regression'] = joblib.load('models/logistic_regression.joblib')
        models['SVM'] = joblib.load('models/svm.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
    except Exception as e:
        return None, None, None, str(e)
    
    try:
        df = pd.read_csv('data/Phishing_Legitimate_full.csv')
    except:
        df = None
        
    return models, scaler, feature_names, df

# Rejection Mechanism
def classify_with_rejection(prob_legitimate, tau_u=0.7, tau_l=0.3):
    if prob_legitimate > tau_u:
        return "LEGITIMATE", prob_legitimate, "Accepted"
    elif prob_legitimate < tau_l:
        return "PHISHING", (1 - prob_legitimate), "Accepted"
    else:
        return "ABSTAIN", max(prob_legitimate, 1-prob_legitimate), "Rejected (Manual Review Required)"

# Log history
def log_history(url_id, model_name, label, confidence, status):
    conn = sqlite3.connect('phishing_detector.db')
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
               str(url_id), model_name, label, confidence, status))
    conn.commit()
    conn.close()

def inject_custom_css():
    st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }

    /* Input text boxes */
    .stTextInput>div>div>input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    /* Select box dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }

    /* Primary buttons */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
    }

    /* Secondary buttons */
    .stButton>button[kind="secondary"] {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button[kind="secondary"]:hover {
        border-color: #94A3B8 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #1E293B !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #38BDF8 !important;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
