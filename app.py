import streamlit as st
from utils import inject_custom_css, init_db, load_resources

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject the premium CSS
inject_custom_css()

# Initialize Database
init_db()

# Ensure session state variables for selective rejection thresholds
if 'tau_u' not in st.session_state:
    st.session_state.tau_u = 0.70
if 'tau_l' not in st.session_state:
    st.session_state.tau_l = 0.30

# Load models and store them globally
models, scaler, feature_names, df = load_resources()

st.title("🛡️ URL Phishing Detector")
st.markdown("### Welcome to the Selective Rejection Phishing Detection System")
st.write("Use the navigation on the left to classify individual URLs, process batches, or explore model performance.")

if not models:
    st.error("⚠️ Machine Learning models are not loaded. Did you run `python train_models.py`?")
else:
    st.success("✅ Models and database loaded successfully. System is ready.")

st.markdown("""
#### Available Features:
- **🔍 Single URL Classifier**: Check a specific URL and extract simulated features in real-time.
- **📊 Batch Analysis**: Upload a CSV to process thousands of URLs at once with selective rejection.
- **📈 Model Dashboard**: View the research metrics (ROC curves, confusion matrices).
- **📜 History**: Browse previously scanned URLs and their manual review statuses.
""")

st.sidebar.markdown("### ⚙️ System Settings")
st.session_state.tau_u = st.sidebar.slider("Upper Threshold (τ_u) [Accept Legitimate]", min_value=0.5, max_value=0.99, value=st.session_state.tau_u, step=0.01)
st.session_state.tau_l = st.sidebar.slider("Lower Threshold (τ_l) [Reject Phishing]", min_value=0.01, max_value=0.5, value=st.session_state.tau_l, step=0.01)

st.sidebar.info("Adjust the thresholds here. Any prediction with confidence between τ_l and τ_u will be flagged for Manual Review.")
