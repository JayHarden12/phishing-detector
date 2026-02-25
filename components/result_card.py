import streamlit as st
from models.enums import PredictionType

def render_result_card(result: dict):
    if "error" in result:
        st.error(result["error"])
        return
        
    prediction = result['prediction']
    confidence = result.get('confidence')
    
    if prediction == PredictionType.PHISHING:
        color = "#DC2626"
        icon = "🚨"
        title = "PHISHING DETECTED"
    elif prediction == PredictionType.LEGITIMATE:
        color = "#16A34A"
        icon = "✅"
        title = "LIKELY SAFE"
    else:
        color = "#F59E0B"
        icon = "⚠️"
        title = "UNCERTAIN - MANUAL REVIEW"
    
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
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem;">{result.get('classifier', '')}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Confidence</p>
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem;">{f"{confidence:.1%}" if confidence else 'N/A'}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Probability</p>
                <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 1.125rem; color: {color};">{result.get('phishing_probability', 0):.1%}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if result.get('rejected'):
        st.info(f"**Rejection Reason:** {result.get('rejection_reason')}. The model's confidence was between the defined thresholds.")

    with st.expander("View Extracted Features"):
        st.json(result.get('features', {}))
