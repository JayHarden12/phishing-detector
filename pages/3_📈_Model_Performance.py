import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.title("📈 Model Performance Dashboard")

st.markdown("Metrics evaluated against a 20% holdout test set.")

try:
    with open("ml_models/metrics.json", "r") as f:
        metrics = json.load(f)
        
    classifier = st.selectbox("Select Classifier to View", [
        "Logistic Regression", "SVM", "Naive Bayes"
    ])
    
    key_map = {
        "Logistic Regression": "logistic_regression",
        "SVM": "svm",
        "Naive Bayes": "naive_bayes"
    }
    
    c_key = key_map[classifier]
    
    class_metrics = next(c for c in metrics['classifiers'] if c['name'] == c_key)
    
    # 1. Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{class_metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{class_metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{class_metrics['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{class_metrics['f1_score']:.2%}")
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Ensure we have data
        cm = class_metrics.get('confusion_matrix', [[0,0],[0,0]])
        # cm is usually [[TN, FP], [FN, TP]]
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Legitimate (0)', 'Phishing (1)'],
                           y=['Legitimate (0)', 'Phishing (1)'],
                           color_continuous_scale='Blues')
        fig_cm.update_xaxes(side="bottom")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.subheader("Feature Importance")
        feats = class_metrics.get('feature_importance')
        if feats:
            # sort by absolute value
            sorted_feats = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            x_vals = [abs(v) for k, v in sorted_feats]
            y_vals = [k for k, v in sorted_feats]
            
            fig_imp = px.bar(x=x_vals, y=y_vals, orientation='h', 
                             title="Top 15 Features by Absolute Weight",
                             labels={'x': 'Absolute Weight', 'y': 'Feature'})
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {classifier}.")

except Exception as e:
    st.error(f"Error loading metrics. Did you run the training script yet? ({e})")
