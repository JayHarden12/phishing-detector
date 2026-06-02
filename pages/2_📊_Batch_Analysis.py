import streamlit as st
import pandas as pd
from utils import load_resources, classify_with_rejection, log_history

st.set_page_config(page_title="Batch Analysis", page_icon="📊", layout="wide")

st.title("📊 Batch URL Analysis")
st.markdown("Upload a CSV file to process multiple URLs at once. The dataset must contain the 48 features used in training.")

models, scaler, feature_names, _ = load_resources()

if models is None:
    st.error("Models not loaded. Please return to the Home page and ensure the models are trained.")
    st.stop()

selected_model = st.selectbox("Select Classifier for Batch:", ["Ensemble (All Models)", "Naive Bayes", "Logistic Regression", "SVM"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** records.")
        
        # Check if all required features exist
        missing_features = [f for f in feature_names if f not in batch_df.columns]
        
        if missing_features:
            st.error(f"Missing {len(missing_features)} required features in the uploaded CSV (e.g., {missing_features[0]}).")
        else:
            if st.button("Run Batch Classification", type="primary"):
                with st.spinner("Processing batch..."):
                    # Extract features
                    X_batch = batch_df[feature_names].values
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    if selected_model == "Ensemble (All Models)":
                        prob_sum = np.zeros(len(X_batch))
                        for model_name in ["Naive Bayes", "Logistic Regression", "SVM"]:
                            prob_sum += models[model_name].predict_proba(X_batch_scaled)[:, 1]
                        probabilities = prob_sum / 3.0
                    else:
                        model = models[selected_model]
                        probabilities = model.predict_proba(X_batch_scaled)[:, 1] # Prob(Legitimate)
                    
                    results = []
                    
                    tau_u = st.session_state.get('tau_u', 0.7)
                    tau_l = st.session_state.get('tau_l', 0.3)
                    
                    for i, prob_legit in enumerate(probabilities):
                        url_id = str(batch_df['id'].iloc[i]) if 'id' in batch_df.columns else f"batch_{i}"
                        
                        label, conf, status = classify_with_rejection(prob_legit, tau_u, tau_l)
                        
                        results.append({
                            "ID/URL": url_id,
                            "Probability (Legit)": prob_legit,
                            "Prediction": label,
                            "Confidence": conf,
                            "Status": status
                        })
                        
                        # Only log a sample to avoid DB lock on massive files
                        if i < 100:
                            log_history(url_id, selected_model, label, conf, status)
                            
                    results_df = pd.DataFrame(results)
                    
                    st.success("Batch processing complete!")
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Processed", len(results_df))
                    
                    auto_accepted = len(results_df[results_df['Status'] == 'Accepted'])
                    col2.metric("Auto-Accepted (Coverage)", f"{auto_accepted} ({auto_accepted/len(results_df):.1%})")
                    
                    manual = len(results_df[results_df['Status'].str.startswith('Rejected')])
                    col3.metric("Manual Review (Rejected)", f"{manual} ({manual/len(results_df):.1%})")
                    
                    # Show dataframe
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name='batch_classification_results.csv',
                        mime='text/csv',
                    )
    except Exception as e:
        st.error(f"Error processing file: {e}")
