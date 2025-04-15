import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import os

# Load the models with better error handling
@st.cache_resource
def load_models():
    try:
        model = load("svm_churn_model2.joblib")
        scaler = load("scaler2.joblib")
        pca_model = load("pca_reduce2.joblib")
        return model, scaler, pca_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please check if model files exist in the repository.")
        # List files in current directory for debugging
        st.write("Files in current directory:", os.listdir())
        return None, None, None

def transform_categorical(df):
    df_encoded = df.copy()
    
    # Building of new features
    df_encoded["Gender_Male"] = (df_encoded["Gender"] == "Male").astype(float)
    df_encoded["Contract Length_Monthly"] = (df_encoded["Contract Length"] == "Monthly").astype(float)
    df_encoded["Contract Length_Quarterly"] = (df_encoded["Contract Length"] == "Quarterly").astype(float)

    # Dropping previous feature
    df_encoded.drop(["Contract Length"], axis=1, inplace=True)

    # Reordering columns to fit the model correctly
    order_of_columns = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 
                       'Gender_Male', 'Contract Length_Monthly', 'Contract Length_Quarterly']
    
    # Ensure all required columns exist
    for col in order_of_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0
            
    df_encoded = df_encoded[order_of_columns]
    
    return df_encoded

def predict_churn_pca(model, scaler, pca_model, data):
    if model is None or scaler is None or pca_model is None:
        return None
        
    try:
        scaled_data = scaler.transform(data)
        pca_data = pca_model.transform(scaled_data)
        prediction = model.predict(pca_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Try restarting the app or check for model compatibility issues.")
        return None

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔄",
    layout="wide"
)

# Title
st.title("Customer Churn Prediction")
st.markdown("""
This application predicts whether a customer is likely to churn based on various factors.
Please fill in the customer information below to get a prediction.
""")

# Load models
model, scaler, pca_model = load_models()

# Display a message if models failed to load
if model is None:
    st.warning("⚠️ Models failed to load. Some features may not work correctly.")

# Create form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        support_calls = st.number_input("Number of Support Calls", min_value=0, value=0)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=0)
    
    with col2:
        contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=100.0)
        last_interaction = st.number_input("Days since Last Interaction", min_value=0, value=0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create DataFrame from input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Contract Length': contract_length,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }])

    # Transform and predict
    transformed_data = transform_categorical(input_data)
    prediction = predict_churn_pca(model, scaler, pca_model, transformed_data)

    # Display prediction
    st.header("Prediction Result")
    if prediction is not None:
        if prediction == 1:
            st.error("⚠️ High Risk of Churn")
            st.markdown("""
            ### Recommendations:
            - Consider reaching out to the customer proactively
            - Review their support history and address any ongoing issues
            - Offer personalized retention deals or upgrades
            """)
        else:
            st.success("✅ Low Risk of Churn")
            st.markdown("""
            ### Recommendations:
            - Continue maintaining good service quality
            - Consider upsell opportunities
            - Monitor for any changes in behavior
            """)
    else:
        st.error("Unable to make prediction. Please check the logs for details.")

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    This churn prediction model uses Support Vector Machine (SVM) algorithm with the following features:
    - Customer demographics (Age, Gender)
    - Behavioral metrics (Support Calls, Payment Delay)
    - Contract information (Length, Total Spend)
    - Engagement metrics (Last Interaction)
    
    The model uses PCA for dimensionality reduction and has been trained on historical customer data.
    """) 