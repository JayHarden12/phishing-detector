import streamlit as st
import json
import os
from services.classification import ClassificationService

st.set_page_config(
    page_title="URL Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Session State
if 'threshold_config' not in st.session_state:
    st.session_state.threshold_config = {
        'naive_bayes': {'upper': 0.7, 'lower': 0.35},
        'svm': {'upper': 0.7, 'lower': 0.35},
        'logistic_regression': {'upper': 0.7, 'lower': 0.35}
    }

if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []

@st.cache_resource(show_spinner=False)
def get_classifier_service():
    return ClassificationService()

def main():
    st.cache_resource.clear()
    
    # Inject Custom Global CSS
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
    st.title("🛡️ URL Phishing Detector")
    st.markdown("### Welcome to the Phishing Detection System")
    st.write("Use the navigation on the left to classify individual URLs, process batches, or configure system settings.")
    
    service = get_classifier_service()
    if not service.is_loaded:
        st.error("⚠️ Machine Learning models are not loaded. Did you run `python train_models.py`?")
    else:
        st.success("✅ Models loaded successfully. System is ready.")
        
    st.markdown("""
    #### Features:
    - **Single URL Classification**: Check a specific URL against our trained models in real-time.
    - **Batch Classification**: Upload a CSV of URLs.
    - **Model Performance**: View training metrics (Accuracy, ROC-AUC, feature importance).
    - **Risk-Coverage**: Configure selective rejection thresholds.
    """)

if __name__ == "__main__":
    main()

