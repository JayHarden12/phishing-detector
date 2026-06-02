import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from utils import load_resources, classify_with_rejection, log_history
from feature_extractor import FeatureExtractor

# Initialize layout
st.set_page_config(page_title="Bulk URL Scanner", page_icon="📂", layout="wide")

st.title("📂 Bulk URL Scanner")
st.write("Upload a CSV or TXT file containing a list of URLs to scan them in bulk.")

models, scaler, feature_names, _ = load_resources()

# Determine standard model to use for bulk to save time
selected_model = "Ensemble (All Models)"

uploaded_file = st.file_uploader("Choose a file (.txt or .csv)", type=['txt', 'csv'])

if uploaded_file is not None:
    # Read the file
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file, header=None)
            urls = df_input[0].tolist()
        else:
            urls = [line.decode('utf-8').strip() for line in uploaded_file.readlines() if line.strip()]
            
        st.info(f"Found {len(urls)} URLs in the uploaded file.")
        
        if st.button("Start Bulk Scan", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            extractor = FeatureExtractor(feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names)
            
            results = []
            
            for i, url in enumerate(urls):
                status_text.text(f"Scanning ({i+1}/{len(urls)}): {url}")
                
                # Extract and scale
                raw_features = extractor.extract_features(url)
                scaled_features = scaler.transform(raw_features)
                
                # Predict
                if selected_model == "Ensemble (All Models)":
                    prediction_label, confidence, status = classify_with_rejection(
                        models['Ensemble'], scaled_features, tau_lower=0.4, tau_upper=0.6)
                else:
                    prediction_label, confidence, status = classify_with_rejection(
                        models[selected_model], scaled_features, tau_lower=0.4, tau_upper=0.6)
                
                results.append({
                    "URL": url,
                    "Prediction": prediction_label,
                    "Confidence": f"{confidence:.1%}" if confidence else "N/A",
                    "Status": status
                })
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Bulk scan complete!")
            
            # Display results
            results_df = pd.DataFrame(results)
            
            # Color code
            def color_rows(row):
                if row['Prediction'] == 'PHISHING':
                    return ['background-color: rgba(220, 38, 38, 0.2)'] * len(row)
                return [''] * len(row)
                
            st.dataframe(results_df.style.apply(color_rows, axis=1), use_container_width=True)
            
            # Provide CSV download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="bulk_scan_results.csv",
                mime="text/csv",
                type="primary"
            )
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
