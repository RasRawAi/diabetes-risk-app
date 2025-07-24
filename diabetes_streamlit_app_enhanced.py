
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model
model = pickle.load(open("xgb_diabetes_model.pkl", "rb"))

# App configuration
st.set_page_config(page_title="Diabetes Risk Screening Tool", layout="centered")
st.title("🩺 Diabetes Risk Screening Tool")

st.markdown("""
This tool estimates an individual's risk of developing diabetes using lifestyle, health, and demographic indicators.
It supports early intervention and healthcare planning. Input your details below to check your risk level.
""")

# BMI Calculator Section
st.subheader("📐 Calculate Your BMI")
height_cm = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight_kg = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
st.success(f"Your calculated BMI is: {bmi}")

# Age Group Details
age_group_labels = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+"
}

# Input Form
st.subheader("📝 Input Your Health Information")
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Age Group", options=list(age_group_labels.keys()), format_func=lambda x: f"{age_group_labels[x]} (Code {x})")
        phys_activity = st.radio("Physically Active?", ["Yes", "No"])
        genhlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        smoker = st.radio("Do you smoke?", ["Yes", "No"])
    with col2:
        highbp = st.radio("High Blood Pressure?", ["Yes", "No"])
        highchol = st.radio("High Cholesterol?", ["Yes", "No"])

    submit = st.form_submit_button("✅ Check My Diabetes Risk")

# On submit
if submit:
    # Map Yes/No to 1/0
    def yn(x): return 1 if x == "Yes" else 0

    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]

    input_df = pd.DataFrame([[
        yn(highbp), yn(highchol), 1, bmi, yn(smoker), 0, 0, yn(phys_activity), 1, 1,
        0, 1, 0, genhlth, 0, 0, 0, 0, age, 5, 6
    ]], columns=feature_names)

    # Predict risk
    risk = model.predict_proba(input_df)[0][1]
    st.subheader("📊 Diabetes Risk Result")
    st.metric(label="Predicted Diabetes Risk Score", value=f"{risk:.2f}")

    if risk > 0.5:
        st.error("⚠️ You are at high risk. Please consult your healthcare provider.")
    else:
        st.success("✅ Your risk is low. Keep maintaining a healthy lifestyle!")

    # Visual Interpretations
    st.subheader("📈 Visual Breakdown")
    fig, ax = plt.subplots()
    ax.pie([1-risk, risk], labels=["No Diabetes", "At Risk"], autopct='%1.1f%%',
           colors=["#A8DADC", "#E76F51"], startangle=90)
    ax.set_title("Diabetes Risk Distribution")
    ax.axis('equal')
    st.pyplot(fig)

    # Bar chart of contributing factors
    st.subheader("📊 Contributing Factors Overview")
    features_display = ['BMI', 'Age', 'Smoker', 'Physical Activity', 'General Health']
    values = [bmi, age, yn(smoker), yn(phys_activity), genhlth]
    bar_colors = ["#457B9D", "#1D3557", "#E76F51", "#2A9D8F", "#F4A261"]

    fig2, ax2 = plt.subplots()
    sns.barplot(x=features_display, y=values, palette=bar_colors, ax=ax2)
    ax2.set_ylabel("Relative Scale / Group Code")
    ax2.set_title("Your Health Profile Summary")
    st.pyplot(fig2)

    # Line graph comparing general health to risk
    st.subheader("📉 Risk vs. General Health Example")
    demo_df = pd.DataFrame({
        "General Health Rating": [1, 2, 3, 4, 5],
        "Estimated Average Risk": [0.1, 0.2, 0.4, 0.6, 0.75]
    })
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=demo_df, x="General Health Rating", y="Estimated Average Risk", marker="o", color="#264653", ax=ax3)
    ax3.set_title("General Health vs. Diabetes Risk (Example Trend)")
    st.pyplot(fig3)

    st.markdown("### 🧠 Model Used: XGBoost Classifier\nTrained on health survey data to predict diabetes risk with high accuracy.")

st.markdown("---")
st.caption("Developed for educational and preventive awareness purposes. Data source: public health indicators.")
