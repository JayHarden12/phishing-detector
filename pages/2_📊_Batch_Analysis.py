import streamlit as st
import pandas as pd
from app import get_classifier_service
from components.result_card import render_result_card
from models.enums import PredictionType
import asyncio

st.title("📊 Batch Analysis")

service = get_classifier_service()

uploaded_file = st.file_uploader("Upload CSV containing a 'url' column", type=["csv"])

classifier_choice = st.selectbox("Classifier for Batch", [
    "Logistic Regression", "SVM", "Naive Bayes", "All (Ensemble)"
])

classifier_map = {
    "All (Ensemble)": "all",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Naive Bayes": "naive_bayes"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'url' not in df.columns:
        # try lowercase match
        url_cols = [c for c in df.columns if c.lower() == 'url']
        if not url_cols:
            st.error("CSV must contain a 'url' column.")
            st.stop()
        else:
            df.rename(columns={url_cols[0]: 'url'}, inplace=True)
            
    st.write(f"Loaded {len(df)} URLs.")
    
    if st.button("Process Batch", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Drop empty rows and ensure strings
        df['url'] = df['url'].astype(str).str.strip()
        df = df[df['url'].astype(bool) & (df['url'] != 'nan')]
        
        results = []
        urls = df['url'].tolist()
        total = len(urls)
        
        # Taking a smaller sample if very large to prevent browser hang
        if total > 500:
            st.warning("Limiting to first 500 rows to prevent timeout issues.")
            urls = urls[:500]
            total = 500
            
        classifier_key = classifier_map[classifier_choice]
        
        for i, url in enumerate(urls):
            # rudimentary normalizer
            if not url.startswith('http'):
                url = 'http://' + url
                
            res = service.classify(url, classifier_key, st.session_state.threshold_config)
            res['url'] = url
            results.append(res)
            
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Processed {i + 1} of {total}")
            
        st.success("Batch Processing Complete!")
        
        # Summary
        valid_results = [r for r in results if "error" not in r]
        errors = len(results) - len(valid_results)
        
        phish = sum(1 for r in valid_results if r['prediction'] == PredictionType.PHISHING)
        legit = sum(1 for r in valid_results if r['prediction'] == PredictionType.LEGITIMATE)
        rejected = sum(1 for r in valid_results if r['prediction'] == PredictionType.REJECTED)
        
        st.write(f"**Phishing:** {phish} | **Legitimate:** {legit} | **Rejected:** {rejected} | **Errors:** {errors}")
        
        if errors > 0:
            st.error("Some evaluations failed. Ensure models are trained and check inputs.")
            
        # Results table
        res_df = pd.DataFrame([{
            'URL': r.get('url', ''),
            'Prediction': r.get('prediction', r.get('error', 'Unknown Error')),
            'Confidence': f"{r.get('confidence', 0):.2f}" if r.get('confidence') else "N/A"
        } for r in results])
        
        st.dataframe(res_df)
