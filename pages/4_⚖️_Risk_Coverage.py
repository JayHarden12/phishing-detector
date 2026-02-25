import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.title("⚖️ Risk-Coverage Configuration")

st.markdown("""
Adjust the thresholds for selective rejection.
- URLs with phishing probability **above** the upper threshold are accepted as Phishing.
- URLs with phishing probability **below** the lower threshold are accepted as Legitimate.
- Anything in between is **rejected** (abstained) for manual review.
""")

classifier = st.selectbox("Select Classifier to Configure", [
    "Logistic Regression", "SVM", "Naive Bayes"
])

key_map = {
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Naive Bayes": "naive_bayes"
}

c_key = key_map[classifier]

col1, col2 = st.columns(2)

current_upper = st.session_state.threshold_config[c_key]['upper']
current_lower = st.session_state.threshold_config[c_key]['lower']

with col1:
    new_upper = st.slider("Upper Threshold (τ_upper)", min_value=0.5, max_value=0.99, value=current_upper, step=0.01)

with col2:
    new_lower = st.slider("Lower Threshold (τ_lower)", min_value=0.01, max_value=0.5, value=current_lower, step=0.01)

if st.button("Save Settings", type="primary"):
    st.session_state.threshold_config[c_key]['upper'] = new_upper
    st.session_state.threshold_config[c_key]['lower'] = new_lower
    st.success("Settings saved successfully!")

# Try to load risk-coverage curve from metrics JSON
try:
    with open("ml_models/metrics.json", "r") as f:
        metrics = json.load(f)
        
    class_metrics = next(c for c in metrics['classifiers'] if c['name'] == c_key)
    curve_data = class_metrics['risk_coverage_curve']
    
    thresholds = [d['threshold'] for d in curve_data]
    coverages = [d['coverage'] for d in curve_data]
    risks = [d['risk'] for d in curve_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coverages, y=risks, mode='lines+markers', name='Risk vs Coverage'))
    
    fig.update_layout(
        title=f"Risk-Coverage Curve ({classifier})",
        xaxis_title="Coverage",
        yaxis_title="Selection Risk (Error Rate)",
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.info("Train models first to see the Risk-Coverage Curve.")
