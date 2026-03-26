import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="AI Doctor", page_icon="🩺")

st.title("🩺 AI Doctor - Disease Risk Prediction System")
st.write("Enter patient details to predict possible disease risks.")

# ---------------- HELPER FUNCTION ---------------- #

def prepare_input(input_list, model):
    required = model.n_features_in_
    while len(input_list) < required:
        input_list.append(0)
    return np.array([input_list])

# ---------------- LOAD MODELS ---------------- #

diabetes_model = joblib.load("diabetes_model.pkl")
lung_model = joblib.load("survey lung cancer.pkl")
thyroid_model = joblib.load("Thyroid_Diff_model.pkl")
alz_model = joblib.load("alzheimers_disease_data_model.pkl")
parkinson_model = joblib.load("parkinson.pkl")
stroke_model = joblib.load("stroke_model_xgb (2).pkl")

# ---------------- SELECT DISEASE ---------------- #

disease = st.selectbox(
    "Select Disease to Check",
    ["Diabetes", "Stroke", "Lung Cancer", "Parkinson", "Thyroid", "Alzheimer"]
)

# ---------------- DIABETES ---------------- #

if disease == "Diabetes":

    st.subheader("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    pedigree = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes Risk"):

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])

        prob = diabetes_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")

# ---------------- STROKE ---------------- #

elif disease == "Stroke":

    st.subheader("Stroke Prediction")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension", [0,1])
    heart = st.selectbox("Heart Disease", [0,1])
    bmi = st.number_input("BMI")
    glucose = st.number_input("Average Glucose Level")

    if st.button("Predict Stroke Risk"):

        input_list = [age, hypertension, heart, glucose, bmi]
        input_data = prepare_input(input_list, stroke_model)

        prob = stroke_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")

# ---------------- LUNG CANCER ---------------- #

elif disease == "Lung Cancer":

    st.subheader("Lung Cancer Prediction")

    age = st.number_input("Age")
    smoking = st.selectbox("Smoking", [0,1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0,1])
    anxiety = st.selectbox("Anxiety", [0,1])
    peer_pressure = st.selectbox("Peer Pressure", [0,1])

    if st.button("Predict Lung Cancer Risk"):

        input_list = [age, smoking, yellow_fingers, anxiety, peer_pressure]
        input_data = prepare_input(input_list, lung_model)

        prob = lung_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")

# ---------------- PARKINSON ---------------- #

elif disease == "Parkinson":

    st.subheader("Parkinson Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("Jitter")

    if st.button("Predict Parkinson Risk"):

        input_list = [fo, fhi, flo, jitter]
        input_data = prepare_input(input_list, parkinson_model)

        prob = parkinson_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")

# ---------------- THYROID ---------------- #

elif disease == "Thyroid":

    st.subheader("Thyroid Prediction")

    age = st.number_input("Age")
    tsh = st.number_input("TSH")
    t3 = st.number_input("T3")
    t4 = st.number_input("T4")

    if st.button("Predict Thyroid Risk"):

        input_list = [age, tsh, t3, t4]
        input_data = prepare_input(input_list, thyroid_model)

        prob = thyroid_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")

# ---------------- ALZHEIMER ---------------- #

elif disease == "Alzheimer":

    st.subheader("Alzheimer Prediction")

    age = st.number_input("Age")
    mmse = st.number_input("MMSE Score")
    cdr = st.number_input("CDR Score")

    if st.button("Predict Alzheimer Risk"):

        input_list = [age, mmse, cdr]
        input_data = prepare_input(input_list, alz_model)

        prob = alz_model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
        elif prob < 0.7:
            st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
        else:
            st.error(f"🔴 High Risk ({prob*100:.1f}%)")
