
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load trained model
model = pickle.load(open("xgb_diabetes_model.pkl", "rb"))

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Screening Tool")

st.markdown("""
This interactive tool helps estimate an individual's risk of developing diabetes 
based on lifestyle and health indicators. It uses a machine learning model trained 
on real-world data and is designed to support early screening and stakeholder awareness.
""")

# Sidebar with stakeholder insights
with st.sidebar:
    st.header("📊 Stakeholder Insights")
    st.markdown("""
    - **Target Audience**: Healthcare providers, policy makers, patients
    - **Use Case**: Quick risk screening and early detection
    - **Model Used**: XGBoost Classifier
    - **Performance**: ~90% AUC with high precision on hold-out test set
    - **How to Use**: Adjust parameters on the right, then click "Check Risk"
    """)

st.subheader("Input Your Health Indicators")

# Input form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        age = st.slider("Age Group (coded 1-13)", 1, 13, 8)
        phys_activity = st.radio("Physically Active?", [0, 1])
        genhlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with col2:
        smoker = st.radio("Smoker?", [0, 1])
        highbp = st.radio("High Blood Pressure?", [0, 1])
        highchol = st.radio("High Cholesterol?", [0, 1])

    submitted = st.form_submit_button("✅ Check Risk")

# Prediction logic
if submitted:
    features = np.array([[highbp, highchol, 1, bmi, smoker, 0, 0, phys_activity, 1, 1,
                          0, 1, 0, genhlth, 0, 0, 0, 0, 0, age, 5, 6]])
    risk = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")
    st.metric(label="Predicted Diabetes Risk Score", value=f"{risk:.2f}")
    
    if risk > 0.5:
        st.warning("⚠️ High Risk – It is advisable to consult a healthcare provider.")
    else:
        st.success("✅ Low Risk – Maintain your healthy lifestyle!")

    # Show visual context
    st.subheader("📈 Model Interpretation & Visuals")
    labels = ['No Diabetes', 'At Risk']
    values = [1 - risk, risk]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#76c7c0', '#e37777'], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Feature influence explanation
    st.markdown("""
    ### 🔍 Why These Features?
    The model considers:
    - **BMI** and **General Health** as key indicators of metabolic stress
    - **Age** and **Smoking** as demographic risk amplifiers
    - **High Blood Pressure** and **Cholesterol** as physiological red flags
    """)

st.markdown("---")
st.caption("Developed as part of a Data Science & AI Capstone Project using open health survey data.")
