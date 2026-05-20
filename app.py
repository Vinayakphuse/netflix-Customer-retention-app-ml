import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Netflix Churn Predictor", layout="wide", page_icon="🎬")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #E50914;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #b20710;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Customer Churn Predictor")
st.markdown("Predict customer churn probability based on streaming habits and demographics. Powered by XGBoost & SHAP.")

API_URL = "http://localhost:8000/predict"

# Sidebar for inputs
st.sidebar.header("User Profile Inputs")

with st.sidebar.expander("👤 Demographics", expanded=True):
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    region = st.selectbox("Region", ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"])

with st.sidebar.expander("💳 Subscription Details", expanded=True):
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    monthly_fee = st.number_input("Monthly Fee ($)", min_value=0.0, value=15.99)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"])
    number_of_profiles = st.slider("Number of Profiles", 1, 5, 2)

with st.sidebar.expander("📺 Streaming Habits", expanded=True):
    device = st.selectbox("Primary Device", ["TV", "Mobile", "Laptop", "Desktop", "Tablet"])
    watch_hours = st.slider("Total Watch Hours (Monthly)", 0.0, 300.0, 50.0)
    avg_watch_time_per_day = st.slider("Avg Watch Time / Day (Hours)", 0.0, 24.0, 2.0)
    last_login_days = st.slider("Days Since Last Login", 0, 60, 2)
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Documentary", "Horror"])

if st.sidebar.button("Predict Churn"):
    # Prepare payload
    payload = {
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "region": region,
        "device": device,
        "monthly_fee": monthly_fee,
        "payment_method": payment_method,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "favorite_genre": favorite_genre
    }
    
    with st.spinner("Analyzing streaming profile..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Display Results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Prediction Result")
                prob = result['churn_probability'] * 100
                will_churn = result['will_churn']
                
                if will_churn:
                    st.error(f"High Risk of Churn! ({prob:.1f}%)")
                    st.markdown("""
                        <div class="metric-card" style="border-left: 5px solid #E50914;">
                            <h4>Action Recommended</h4>
                            <p>Target with immediate personalized retention offer or content recommendation.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"Low Risk of Churn ({prob:.1f}%)")
                    st.markdown("""
                        <div class="metric-card" style="border-left: 5px solid #2e7d32;">
                            <h4>Action Recommended</h4>
                            <p>Customer is highly engaged. Continue standard service.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                st.subheader("SHAP Explainability (Why?)")
                st.markdown("Positive values push the prediction towards **Churn**, negative values push towards **Retention**.")
                
                shap_values = result['shap_values']
                
                # Plot SHAP waterfall/bar chart
                features = list(shap_values.keys())
                impacts = list(shap_values.values())
                
                # Sort by absolute impact
                sorted_idx = np.argsort(np.abs(impacts))
                sorted_features = [features[i] for i in sorted_idx]
                sorted_impacts = [impacts[i] for i in sorted_idx]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                # Custom colors for Netflix theme
                colors = ['#E50914' if val > 0 else '#221f1f' for val in sorted_impacts]
                ax.barh(sorted_features, sorted_impacts, color=colors)
                ax.set_xlabel("SHAP Value (Impact on Log-Odds of Churn)")
                ax.set_title("Feature Contributions to Prediction")
                ax.axvline(0, color='gray', linewidth=1, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("### Business Impact Summary")
st.markdown("This dashboard empowers customer success teams to identify at-risk users in real-time, explain the drivers of churn, and deploy targeted retention strategies efficiently.")
