import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="AI Doctor", page_icon="🩺")
st.title("🩺 AI Doctor - Disease Risk Prediction System")
st.write("Enter patient details to predict possible disease risks.")

# ---------------- Load Diabetes Model ---------------- #
diabetes_model = joblib.load("diabetes_model.pkl")  # Keep real model

# ---------------- Demo probability functions ---------------- #

def demo_prob_stroke(age, hypertension, heart, glucose, bmi, gender, smoking):
    prob = 0.05
    prob += 0.02 * (age / 100)
    prob += 0.1 * hypertension
    prob += 0.1 * heart
    prob += 0.1 * (glucose / 200)
    prob += 0.05 * (bmi / 50)
    prob += 0.05 * gender
    prob += 0.1 * smoking
    return min(prob, 1.0)

def demo_prob_lung(age, smoking, yellow_fingers, anxiety, peer_pressure, alcohol, cough):
    prob = 0.05
    prob += 0.02 * (age / 100)
    prob += 0.15 * smoking
    prob += 0.05 * yellow_fingers
    prob += 0.05 * anxiety
    prob += 0.05 * peer_pressure
    prob += 0.05 * alcohol
    prob += 0.15 * cough
    return min(prob, 1.0)

def demo_prob_parkinson(fo, fhi, flo, jitter, shimmer, hnr):
    prob = 0.05
    prob += 0.001 * (fo / 100)
    prob += 0.001 * (fhi / 100)
    prob += 0.001 * (flo / 100)
    prob += 0.1 * jitter
    prob += 0.1 * shimmer
    prob += 0.05 * (20 - hnr) / 20
    return min(prob, 1.0)

def demo_prob_thyroid(age, tsh, t3, t4, fatigue):
    prob = 0.05
    prob += 0.01 * (age / 100)
    prob += 0.1 * (tsh / 10)
    prob += 0.05 * (1 - t3 / 3)
    prob += 0.05 * (1 - t4 / 12)
    prob += 0.1 * fatigue
    return min(prob, 1.0)

def demo_prob_alzheimer(age, mmse, cdr, memory_loss, confusion):
    prob = 0.05
    prob += 0.02 * (age / 100)
    prob += 0.1 * (30 - mmse) / 30
    prob += 0.1 * (cdr / 3)
    prob += 0.15 * memory_loss
    prob += 0.15 * confusion
    return min(prob, 1.0)

# ---------------- Helper Function ---------------- #
def show_result(prob, threshold=0.35):
    if prob >= threshold:
        st.error(f"🔴 Consult a doctor ({prob*100:.1f}%)")
    else:
        st.success(f"🟢 You are OK ({prob*100:.1f}%)")

# ---------------- Disease Selection ---------------- #
disease = st.selectbox(
    "Select Disease to Check",
    ["Diabetes", "Stroke", "Lung Cancer", "Parkinson", "Thyroid", "Alzheimer"]
)

# ---------------- Diabetes ---------------- #
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
        show_result(prob, threshold=0.4)

# ---------------- Stroke ---------------- #
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
        prob = demo_prob_stroke(age, hypertension, heart, glucose, bmi, gender, smoking)
        show_result(prob, threshold=0.35)

# ---------------- Lung Cancer ---------------- #
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
        prob = demo_prob_lung(age, smoking, yellow_fingers, anxiety, peer_pressure, alcohol, cough)
        show_result(prob, threshold=0.35)

# ---------------- Parkinson ---------------- #
elif disease == "Parkinson":
    st.subheader("Parkinson Prediction")
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("Jitter")
    shimmer = st.number_input("Shimmer")
    hnr = st.number_input("HNR")

    if st.button("Predict Parkinson Risk"):
        prob = demo_prob_parkinson(fo, fhi, flo, jitter, shimmer, hnr)
        show_result(prob, threshold=0.35)

# ---------------- Thyroid ---------------- #
elif disease == "Thyroid":
    st.subheader("Thyroid Prediction")
    age = st.number_input("Age")
    tsh = st.number_input("TSH")
    t3 = st.number_input("T3")
    t4 = st.number_input("T4")
    fatigue = st.selectbox("Fatigue (0=No,1=Yes)", [0,1])

    if st.button("Predict Thyroid Risk"):
        prob = demo_prob_thyroid(age, tsh, t3, t4, fatigue)
        show_result(prob, threshold=0.35)

# ---------------- Alzheimer ---------------- #
elif disease == "Alzheimer":
    st.subheader("Alzheimer Prediction")
    age = st.number_input("Age")
    mmse = st.number_input("MMSE Score")
    cdr = st.number_input("CDR Score")
    memory_loss = st.selectbox("Memory Loss (0=No,1=Yes)", [0,1])
    confusion = st.selectbox("Confusion (0=No,1=Yes)", [0,1])

    if st.button("Predict Alzheimer Risk"):
        prob = demo_prob_alzheimer(age, mmse, cdr, memory_loss, confusion)
        show_result(prob, threshold=0.35)
