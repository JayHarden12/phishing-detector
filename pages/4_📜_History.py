import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Scan History", page_icon="📜", layout="wide")

st.title("📜 URL Scan History")
st.markdown("This database logs all individual URL scans and a sample of batch scans.")

def get_history():
    try:
        conn = sqlite3.connect('phishing_detector.db')
        df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()

df = get_history()

if not df.empty:
    st.write(f"Total records: {len(df)}")
    
    # Filtering
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect("Filter by Status", df['rejection_status'].unique(), default=df['rejection_status'].unique())
    with col2:
        model_filter = st.multiselect("Filter by Model", df['model'].unique(), default=df['model'].unique())
        
    filtered_df = df[df['rejection_status'].isin(status_filter) & df['model'].isin(model_filter)]
    
    # Styled dataframe
    def color_status(val):
        if val.startswith("Rejected"):
            color = "#F59E0B"
        elif val == "Accepted" or val == "Accepted ": # Just in case
            color = "#16A34A"
        else:
            color = "inherit"
        return f'color: {color}'
        
    st.dataframe(filtered_df.style.map(color_status, subset=['rejection_status']), use_container_width=True)
    
    if st.button("Clear History", type="secondary"):
        conn = sqlite3.connect('phishing_detector.db')
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        st.success("History cleared.")
        st.rerun()
else:
    st.info("No scan history found yet. Go to the Single URL Classifier to run some tests!")
