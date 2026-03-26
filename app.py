import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="AI Doctor", page_icon="🩺")

st.title("🩺 AI Doctor - Disease Risk Prediction System")
st.write("Enter patient details to predict possible disease risks.")

# ---------------- HELPER FUNCTIONS ---------------- #

def prepare_input(input_list, model):
    required = model.n_features_in_
    while len(input_list) < required:
        input_list.append(0)
    return np.array([input_list])

def adjust_probability(prob, inputs):
    inputs = np.array(inputs, dtype=float)

    # Normalize inputs safely
    if np.max(inputs) != 0:
        norm_inputs = inputs / np.max(inputs)
    else:
        norm_inputs = inputs

    score = np.mean(norm_inputs)

    # Balanced adjustment
    adjusted = prob * 0.6 + score * 0.4

    return max(0, min(1, adjusted))

def show_result(prob):
    if prob < 0.35:
        st.success(f"🟢 Low Risk ({prob*100:.1f}%)")
    elif prob < 0.65:
        st.warning(f"🟡 Medium Risk ({prob*100:.1f}%)")
    else:
        st.error(f"🔴 High Risk ({prob*100:.1f}%)")

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

        show_result(prob)

# ---------------- STROKE ---------------- #

elif disease == "Stroke":

    st.subheader("Stroke Prediction")

    age = st.number_input("Age")
    hypertension = st.selectbox("Hypertension", [0,1])
    heart = st.selectbox("Heart Disease", [0,1])
    bmi = st.number_input("BMI")
    glucose = st.number_input("Average Glucose Level")
    gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
    smoking = st.selectbox("Smoking (0=No,1=Yes)", [0,1])

    if st.button("Predict Stroke Risk"):

        input_list = [age, hypertension, heart, glucose, bmi, gender, smoking]
        input_data = prepare_input(input_list, stroke_model)

        prob = stroke_model.predict_proba(input_data)[0][1]
        prob = adjust_probability(prob, input_list)

        show_result(prob)

# ---------------- LUNG CANCER ---------------- #

elif disease == "Lung Cancer":

    st.subheader("Lung Cancer Prediction")

    age = st.number_input("Age")
    smoking = st.selectbox("Smoking", [0,1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0,1])
    anxiety = st.selectbox("Anxiety", [0,1])
    peer_pressure = st.selectbox("Peer Pressure", [0,1])
    alcohol = st.selectbox("Alcohol Consumption", [0,1])
    cough = st.selectbox("Chronic Cough", [0,1])

    if st.button("Predict Lung Cancer Risk"):

        input_list = [age, smoking, yellow_fingers, anxiety, peer_pressure, alcohol, cough]
        input_data = prepare_input(input_list, lung_model)

        prob = lung_model.predict_proba(input_data)[0][1]
        prob = adjust_probability(prob, input_list)

        show_result(prob)

# ---------------- PARKINSON ---------------- #

elif disease == "Parkinson":

    st.subheader("Parkinson Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("Jitter")
    shimmer = st.number_input("Shimmer")
    hnr = st.number_input("HNR")

    if st.button("Predict Parkinson Risk"):

        input_list = [fo, fhi, flo, jitter, shimmer, hnr]
        input_data = prepare_input(input_list, parkinson_model)

        prob = parkinson_model.predict_proba(input_data)[0][1]
        prob = adjust_probability(prob, input_list)

        show_result(prob)

# ---------------- THYROID ---------------- #

elif disease == "Thyroid":

    st.subheader("Thyroid Prediction")

    age = st.number_input("Age")
    tsh = st.number_input("TSH")
    t3 = st.number_input("T3")
    t4 = st.number_input("T4")
    fatigue = st.selectbox("Fatigue (0=No,1=Yes)", [0,1])

    if st.button("Predict Thyroid Risk"):

        input_list = [age, tsh, t3, t4, fatigue]
        input_data = prepare_input(input_list, thyroid_model)

        prob = thyroid_model.predict_proba(input_data)[0][1]
        prob = adjust_probability(prob, input_list)

        show_result(prob)

# ---------------- ALZHEIMER ---------------- #

elif disease == "Alzheimer":

    st.subheader("Alzheimer Prediction")

    age = st.number_input("Age")
    mmse = st.number_input("MMSE Score")
    cdr = st.number_input("CDR Score")
    memory_loss = st.selectbox("Memory Loss (0=No,1=Yes)", [0,1])
    confusion = st.selectbox("Confusion (0=No,1=Yes)", [0,1])

    if st.button("Predict Alzheimer Risk"):

        input_list = [age, mmse, cdr, memory_loss, confusion]
        input_data = prepare_input(input_list, alz_model)

        prob = alz_model.predict_proba(input_data)[0][1]
        prob = adjust_probability(prob, input_list)

        show_result(prob)
