import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📉 Customer Churn Predictor")
st.markdown("Use the sliders below to simulate a customer's behavior and predict churn risk.")

# ------------------ Load Artifacts ------------------
try:
    model = joblib.load('xgb_churn_model2.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
except FileNotFoundError:
    st.error("❌ Missing required files. Please run the training script first.")
    st.stop()

# List of features
features =[
    'App Logins', 'Loans Accessed', 'Loans Taken',
    'Sentiment Score', 'Web Logins', 'Monthly Avg Balance',
    'Declined Txns', 'Overdraft Events', 'Tickets Raised'
]

# ------------------ User Inputs ------------------
st.sidebar.header("🎛️ Customer Behavior")

input_data = {}

for feature in features:
    if feature == 'Sentiment Score':
        input_data[feature] = st.sidebar.slider(feature, 0.0, 1.0, 0.5, 0.01)
    elif feature in ['Loan Accessed', 'Declined Tax', 'Overdrafts Event', 'Tickets Raised']:
        input_data[feature] = st.sidebar.slider(feature, 0, 10, 0, 1)
    elif feature == 'Loans Taken':
        input_data[feature] = st.sidebar.slider(feature, 0, 20, 2, 1)
    elif feature == 'Monthly Avg Balance':
        input_data[feature] = st.sidebar.slider(feature, 0, 10000, 2500, 100)
    else:
        input_data[feature] = st.sidebar.slider(feature, 0, 100, 20, 1)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply preprocessing
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

# Predict
prob = model.predict_proba(input_scaled)[0][1]
risk_tier = "Low" if prob < 0.4 else "Medium" if prob < 0.7 else "High"

# Generate recommendation
if risk_tier == "Low":
    rec = ["Send monthly loyalty points"]
elif risk_tier == "Medium":
    rec = ["Offer $5 cashback for next renewal"]
else:
    rec = ["Immediate support call + plan upgrade offer"]

# Add predictions to input data
result_df = input_df.copy()
result_df['Churn_Probability'] = prob
result_df['Risk_Tier'] = risk_tier
result_df['Recommendations'] = str(rec)

# ------------------ Display Results ------------------
st.subheader("📊 Input Values")
st.write(input_df)

st.subheader("🎯 Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{prob:.2%}")

with col2:
    if risk_tier == "Low":
        st.success(f"Risk Tier: {risk_tier}")
    elif risk_tier == "Medium":
        st.warning(f"Risk Tier: {risk_tier}")
    else:
        st.error(f"Risk Tier: {risk_tier}")

with col3:
    st.info("Recommendation:")
    for r in rec:
        st.markdown(f"- {r}")

# Show full result
st.markdown("---")
st.subheader("📥 Full Input & Prediction")
st.dataframe(result_df.style.format({'Churn_Probability': '{:.2%}'}))

# Optional: Download button
csv = result_df.to_csv(index=False).encode()
st.download_button("💾 Download Prediction", csv, "churn_prediction.csv", "text/csv")