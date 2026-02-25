import streamlit as st
import pandas as pd

st.title("📜 Classification History")

st.markdown("Review previously checked URLs from your session.")

history = st.session_state.get('classification_history', [])

if not history:
    st.info("No URLs have been classified yet in this session.")
else:
    # Convert to dataframe for nice display
    df = pd.DataFrame([{
        "Timestamp": r.get('timestamp', ''),
        "URL": r['url'],
        "Classifier": r['classifier'],
        "Prediction": r['prediction'],
        "Confidence": f"{r['confidence']:.2%}" if r.get('confidence') else "N/A",
        "Rejected": "Yes" if r.get('rejected') else "No",
        "Rejection Reason": r.get('rejection_reason', '')
    } for r in history])
    
    # Sort by timestamp descending
    df = df.sort_values('Timestamp', ascending=False)
    
    # Allow filtering by prediction
    filter_pred = st.selectbox("Filter by Prediction", ["All", "phishing", "legitimate", "rejected"])
    if filter_pred != "All":
        df = df[df['Prediction'] == filter_pred]
        
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download as CSV button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name='phishing_classification_history.csv',
        mime='text/csv',
    )
