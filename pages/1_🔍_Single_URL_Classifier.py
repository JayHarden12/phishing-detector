import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import load_resources, classify_with_rejection, log_history
from feature_extractor import FeatureExtractor
from components.result_card import render_result_card, render_explainability_chart

# Initialize layout
st.set_page_config(page_title="Single URL Check", page_icon="🔍", layout="wide")

st.title("🔍 Single URL Classifier")
st.write("Enter a URL below to extract features and classify it in real-time.")

models, scaler, feature_names, _ = load_resources()

if models is None:
    st.error("Models not loaded. Please return to the Home page and ensure the models are trained.")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_input("Enter URL to scan:", placeholder="http://example.com/login")
    
with col2:
    selected_model = st.selectbox("Select Classifier:", ["Ensemble (All Models)", "Naive Bayes", "Logistic Regression", "SVM"])

if st.button("Analyze URL", type="primary"):
    if not url_input:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Extracting features and analyzing..."):
            # Mock feature extraction (for demo purposes)
            # In a real system, you would extract all 48 features from the URL string
            # Instantiate the extractor
            extractor = FeatureExtractor(feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names)
            
            with st.spinner("Extracting features from URL and analyzing content..."):
                # Real feature extraction
                mock_features = extractor.extract_features(url_input)
                
                # We rename it to mock_features just to avoid renaming everything below, 
                # but it's actually real features now!
                mock_features_scaled = scaler.transform(mock_features)
            
            # Predict
            if selected_model == "Ensemble (All Models)":
                # Average probabilities from all 3 models
                prob_sum = 0
                for model_name in ["Naive Bayes", "Logistic Regression", "SVM"]:
                    prob_sum += models[model_name].predict_proba(mock_features_scaled)[0][1]
                prob_legit = prob_sum / 3.0
            else:
                model = models[selected_model]
                prob_legit = model.predict_proba(mock_features_scaled)[0][1]
            
            # Apply selective rejection
            prediction_label, confidence, status = classify_with_rejection(
                prob_legit, 
                st.session_state.get('tau_u', 0.7), 
                st.session_state.get('tau_l', 0.3)
            )
            
            # Log to DB
            log_history(url_input, selected_model, prediction_label, confidence, status)
            
            # Render custom UI card
            render_result_card(prediction_label, confidence, status, selected_model)
            
            # Render Explainable AI Chart
            st.divider()
            lr_model = models.get('Logistic Regression')
            if lr_model is not None:
                render_explainability_chart(mock_features_scaled, feature_names, lr_model)
            
            with st.expander("View Extracted Features"):
                st.json({feature_names[i]: float(mock_features[0][i]) for i in range(len(feature_names))})
