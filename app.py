import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
from catboost import CatBoostClassifier

# Set page config
st.set_page_config(page_title="AML Mule Detection", layout="wide")

st.title("🛡️ AML Mule Account Detection System")
st.markdown("""
This dashboard displays predictions for potentially suspicious accounts.
The system uses an ensemble of CatBoost, XGBoost, and LightGBM models.
""")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load predictions if they exist
@st.cache_data
def load_submissions():
    sub_file = os.path.join(DATA_DIR, "submission_v15.csv")
    if os.path.exists(sub_file):
        return pd.read_csv(sub_file)
    return None

submissions = load_submissions()

if submissions is not None:
    st.header("📊 Prediction Overview")
    
    col1, col2, col3 = st.columns(3)
    total_accounts = len(submissions)
    mules_detected = (submissions['is_mule'] > 0.5).sum()
    
    col1.metric("Total Accounts Scanned", f"{total_accounts:,}")
    col2.metric("Mules Detected (p>0.5)", f"{mules_detected:,}", delta_color="inverse")
    col3.metric("Mule Rate (%)", f"{(mules_detected/total_accounts*100):.2f}%")

    st.subheader("High Risk Accounts")
    high_risk = submissions[submissions['is_mule'] > 0.8].sort_values('is_mule', ascending=False)
    st.dataframe(high_risk, use_container_width=True)

    st.subheader("Score Distribution")
    st.bar_chart(np.histogram(submissions['is_mule'], bins=20)[0])

else:
    st.warning("⚠️ Submission file `submission_v15.csv` not found. Please run the training script first.")

# Sidebar for manual lookup
st.sidebar.header("Account Investigation")
search_id = st.sidebar.text_input("Enter Account ID:")
if search_id and submissions is not None:
    match = submissions[submissions['account_id'] == search_id]
    if not match.empty:
        st.sidebar.success(f"Mule Probability: {match['is_mule'].values[0]:.4%}")
        if match['suspicious_start'].values[0]:
            st.sidebar.info(f"Suspicious Window: \n{match['suspicious_start'].values[0]} to {match['suspicious_end'].values[0]}")
    else:
        st.sidebar.error("Account not found.")
