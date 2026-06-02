import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Model Dashboard", page_icon="📈", layout="wide")

st.title("📈 Model Performance Dashboard")
st.markdown("This dashboard displays the evaluation metrics from the cross-validation and test set evaluations.")

try:
    with open('results/evaluation_summary.json', 'r') as f:
        metrics = json.load(f)
        
    st.subheader("Comparative Test Metrics")
    
    # Create a nice dataframe
    df_metrics = pd.DataFrame(metrics).T
    
    # Format
    st.dataframe(
        df_metrics.style.format("{:.4f}").background_gradient(cmap="Blues"),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curves")
        try:
            st.image("results/roc_curves.png", use_container_width=True)
        except:
            st.info("ROC Curve image not found.")
                
    with col2:
        st.subheader("Confusion Matrices")
        try:
            # We can show all three or let the user pick, but let's just show them sequentially
            st.image("results/confusion_matrix_Ensemble.png", use_container_width=True, caption="Ensemble (Soft Voting)")
            st.image("results/confusion_matrix_Naive_Bayes.png", use_container_width=True, caption="Naive Bayes")
            st.image("results/confusion_matrix_Logistic_Regression.png", use_container_width=True, caption="Logistic Regression")
            st.image("results/confusion_matrix_SVM.png", use_container_width=True, caption="SVM")
        except:
            st.info("Confusion Matrix image not found.")

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Risk-Coverage Curve (Selective Rejection)")
        try:
            st.image("results/risk_coverage.png", use_container_width=True)
        except:
            st.info("Risk-Coverage image not found.")
                 
    with col4:
        st.subheader("Feature Importance")
        try:
            st.image("results/feature_importance.png", use_container_width=True)
        except:
            st.info("Feature importance image not found.")

except FileNotFoundError:
    st.error("Evaluation metrics not found. Did you run `python train_models.py`?")
