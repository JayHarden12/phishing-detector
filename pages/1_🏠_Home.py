import streamlit as st
from components.result_card import render_result_card
import pandas as pd
from datetime import datetime

# We need access to the cached service
from app import get_classifier_service

st.title("🔍 Check URL Safety")

service = get_classifier_service()

url = st.text_input("Enter URL", placeholder="https://example.com")
classifier_choice = st.selectbox("Classifier", [
    "All (Ensemble)", "Logistic Regression", "SVM", "Naive Bayes"
])

classifier_map = {
    "All (Ensemble)": "all",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Naive Bayes": "naive_bayes"
}

if st.button("Check URL", type="primary"):
    if not url:
        st.warning("Please enter a URL")
    else:
        if not url.startswith('http'):
            url = 'http://' + url
            
        with st.spinner("Analyzing..."):
            classifier_key = classifier_map[classifier_choice]
            result = service.classify(url, classifier_key, st.session_state.threshold_config)
            
            # Save to history only if valid prediction
            if "error" not in result:
                record = result.copy()
                record['url'] = url
                record['timestamp'] = datetime.now().isoformat()
                st.session_state.classification_history.append(record)
            
            render_result_card(result)
