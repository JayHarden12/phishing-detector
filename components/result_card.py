import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def render_result_card(prediction_label, confidence_score, status, classifier_name):
    
    if status.startswith("Rejected"):
        color = "#F59E0B" # Yellow/Orange
        icon = "⚠️"
        title = "UNCERTAIN - MANUAL REVIEW REQUIRED"
    elif prediction_label == "PHISHING":
        color = "#DC2626" # Red
        icon = "🚨"
        title = "PHISHING DETECTED"
    else: # LEGITIMATE
        color = "#16A34A" # Green
        icon = "✅"
        title = "LIKELY SAFE"
    
    st.markdown(f"""
    <div style="
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid {color};
        background-color: {color}1A;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        margin-bottom: 24px;
        transition: transform 0.2s ease-in-out;
    ">
        <h3 style="color: {color}; margin-top: 0; margin-bottom: 16px; font-weight: 800; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.5em;">{icon}</span> {title}
        </h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div>
                <p style="margin: 0; font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Classifier</p>
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem;">{classifier_name}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Confidence</p>
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem;">{f"{confidence_score:.1%}" if confidence_score else 'N/A'}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Status</p>
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem; color: {color};">{status}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if status.startswith("Rejected"):
        st.info("The model's confidence was between the defined thresholds (τ_lower and τ_upper). A definitive prediction could not be made securely, so it has been routed for manual review.")

def render_explainability_chart(features_scaled, feature_names, lr_model):
    """
    Renders an Explainable AI (XAI) chart using Logistic Regression weights as a proxy.
    Calculates the contribution of each feature to the final prediction.
    """
    st.markdown("### 🧠 AI Explainability")
    st.write("Top features that influenced this prediction:")
    
    # Calculate contributions: feature_value * model_weight
    weights = lr_model.coef_[0]
    contributions = features_scaled[0] * weights
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions
    })
    
    # Sort by absolute contribution to find the most impactful features
    df['AbsContribution'] = np.abs(df['Contribution'])
    top_features = df.sort_values(by='AbsContribution', ascending=False).head(7)
    
    # Add color column based on whether it pushes toward Phishing (Positive) or Legitimate (Negative)
    top_features['Direction'] = np.where(top_features['Contribution'] > 0, 'Increases Phishing Risk', 'Decreases Phishing Risk')
    
    # Plot using Plotly Express
    fig = px.bar(
        top_features, 
        x='Contribution', 
        y='Feature', 
        orientation='h',
        color='Direction',
        color_discrete_map={
            'Increases Phishing Risk': '#DC2626', # Red
            'Decreases Phishing Risk': '#16A34A'  # Green
        },
        title="Top 7 Influencing Factors (LR Proxy)"
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
