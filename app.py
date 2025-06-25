import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Obesity Prediction", layout="centered")
st.title("🧍‍♂️ Obesity Prediction Form 🧍‍♀️")

gender_options = ["Female", "Male"]
yes_no_options = ["no", "yes"]
caec_calc_options = ["no", "Sometimes", "Frequently", "Always"]
mtrans_options = ["Public_Transportation", "Automobile", "Others"]

with st.form("obesity_form"):
    st.header("👤 Personal Info & Body Measurements")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("🚻 Gender", gender_options)
        age = st.number_input("🎂 Age (years)", 1.0, 100.0, 25.0, 1.0, format="%.1f")
        height = st.number_input("📏 Height (meters)", 1.0, 2.5, 1.70, 0.01, format="%.2f")
        weight = st.number_input("⚖️ Weight (kg)", 10.0, 250.0, 70.0, 0.1, format="%.1f")
        family_history_with_overweight = st.selectbox("👪 Family History With Overweight", yes_no_options)
        favc = st.selectbox("🍔 Frequent High Caloric Food Consumption (FAVC)", yes_no_options)
    with col2:
        fcvc = st.slider("🥦 Frequency of Vegetable Consumption (FCVC)", 1.0, 5.0, 3.0, 0.1)
        ncp = st.slider("🍽 Number of Main Meals per Day (NCP)", 1.0, 5.0, 3.0, 0.1)
        caec = st.selectbox("🍫 Consumption of Food Between Meals (CAEC)", caec_calc_options)
        smoke = st.selectbox("🚬 Do you smoke? (SMOKE)", yes_no_options)
        ch2o = st.slider("💧 Daily Water Consumption (CH2O in liters)", 1.0, 5.0, 2.0, 0.1)
        scc = st.selectbox("📊 Calories Consumption Monitoring (SCC)", yes_no_options)

    st.header("🏃‍♂️ Lifestyle & Habits")
    col3, col4 = st.columns(2)
    with col3:
        faf = st.slider("🏋️ Physical Activity Frequency (FAF) (hours/day)", 0.0, 5.0, 1.0, 0.1)
        tue = st.slider("📱 Time Using Technology (TUE) (hours/day)", 0.0, 10.0, 2.0, 0.1)
    with col4:
        calc = st.selectbox("🍷 Consumption of Alcohol (CALC)", caec_calc_options)
        mtrans = st.selectbox("🚗 Most Used Transportation (MTRANS)", mtrans_options)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history_with_overweight,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    url = "https://deploymentuas-production.up.railway.app/predict"

    try:
        response = requests.post(url, json=input_data)
        response.raise_for_status()
        result = response.json()

        st.subheader("📝 Input Data")
        st.write(pd.DataFrame([input_data]))

        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Obesity Category: **{result['prediction']}**")

        st.subheader("📊 Prediction Probabilities")
        probs = result.get("probabilities", {})
        if probs:
            df_probs = pd.DataFrame.from_dict(probs, orient='index', columns=['Probability'])
            st.bar_chart(df_probs)

        st.subheader("💡 Feature Importances")
        fi = result.get("feature_importances", {})
        if fi:
            df_fi = pd.DataFrame.from_dict(fi, orient='index', columns=['Importance'])
            df_fi = df_fi.sort_values(by='Importance', ascending=False)
            st.bar_chart(df_fi)

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error calling prediction API: {e}")
