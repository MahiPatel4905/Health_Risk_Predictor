import base64
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# OCR IMPORTS
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re

# ✅ Tesseract Path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Health Risk Predictor", layout="centered")


# ================== BACKGROUND IMAGE ==================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
        color: black;
    }}

    input, select, textarea {{
        background-color: #f0f2f6 !important;
        color: black !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
        border: 2px solid transparent !important;
    }}

    div[data-baseweb="select"] * {{
        color: black !important;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        font-size: 18px;
    }}

    h1, h2, h3 {{
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)


set_bg("eg.jpg")


# ================== SESSION ==================
if "page" not in st.session_state:
    st.session_state.page = "welcome"


# ================== OCR FUNCTIONS ==================
def extract_text_from_image(image):
    img = Image.open(image)
    return pytesseract.image_to_string(img)


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_value(pattern, text, default=0):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


# ================== MULTI DISEASE DETECTION ==================
def detect_diseases(text):
    t = text.lower()
    diseases = []

    if "blood pressure" in t or "systolic" in t or "diastolic" in t:
        diseases.append("bp")

    if "diabetes" in t or "glucose" in t or "hba1c" in t:
        diseases.append("dia")

    if "cholesterol" in t or "heart rate" in t or "ecg" in t:
        diseases.append("heart")

    if "bilirubin" in t or "albumin" in t or "liver" in t:
        diseases.append("liver")

    if "lung" in t or "smoking" in t or "chest pain" in t:
        diseases.append("lung")

    return diseases


# ================== MODEL CACHE ==================
@st.cache_resource
def load_models():
    lung_df = pd.read_excel("lungcancer.xlsx")
    dia_df = pd.read_excel("diabetes_prediction_dataset.xlsx")
    heart_df = pd.read_excel("HeartDiseaseTrain-Test.xlsx")
    bp_df = pd.read_excel("hypertension_dataset.xlsx")
    liver_df = pd.read_excel("Indian Liver Patient Dataset (ILPD).xlsx")

    lung_df["LUNG_CANCER"] = lung_df["LUNG_CANCER"].astype(str).str.upper()
    bp_df["Has_Hypertension"] = bp_df["Has_Hypertension"].astype(str).str.upper()

    # LUNG
    lung_X = lung_df[["GENDER","AGE","SMOKING","ALCOHOL_CONSUMING","SHORTNESS_OF_BREATH","CHEST_PAIN"]].copy()
    lung_X["GENDER"] = lung_X["GENDER"].map({"M": 1, "F": 0})
    lung_y = lung_df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
    lung_model = RandomForestClassifier().fit(lung_X, lung_y)

    # DIABETES
    dia_X = dia_df[["gender","age","hypertension","heart_disease","bmi"]].copy()
    dia_X["gender"] = dia_X["gender"].map({"Male": 1, "Female": 0})
    dia_y = dia_df["diabetes"]
    dia_model = RandomForestClassifier().fit(dia_X, dia_y)

    # HEART
    heart_X = heart_df[["age","sex","cholestoral","Max_heart_rate","thalassemia"]].copy()
    heart_X["sex"] = heart_X["sex"].map({"Male": 1, "Female": 0})
    heart_X["thalassemia"] = heart_X["thalassemia"].map({"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2})
    heart_y = heart_df["target"]
    heart_model = RandomForestClassifier().fit(heart_X, heart_y)

    # BP
    bp_X = bp_df[["Age","BMI","Salt_Intake","BP_History","Smoking_Status"]].copy()
    bp_X["BP_History"] = bp_X["BP_History"].map({"Normal": 0, "High": 1})
    bp_X["Smoking_Status"] = bp_X["Smoking_Status"].map({"Non-Smoker": 0, "Smoker": 1})
    bp_y = bp_df["Has_Hypertension"].map({"YES": 1, "NO": 0})
    bp_model = RandomForestClassifier().fit(bp_X, bp_y)

    # LIVER
    liver_X = liver_df[["age","gender","tot_bilirubin","direct_bilirubin","tot_proteins","albumin","ag_ratio"]].copy()
    liver_X["gender"] = liver_X["gender"].map({"Male": 1, "Female": 0})
    liver_y = liver_df["is_patient"]
    liver_model = RandomForestClassifier().fit(liver_X, liver_y)

    return lung_model, dia_model, heart_model, bp_model, liver_model


# ================== WELCOME PAGE ==================
if st.session_state.page == "welcome":

    st.title("💊 Health Risk Predictor")
    st.markdown("### 🧠 Smart AI-Based Health Analysis System")

    st.markdown("""
    <br>
    <div style='text-align:center; font-size:18px;'>
    🔍 Predict multiple health risks <br><br>
    ❤️ Heart | 🩸 Diabetes | 🫁 Lung | 💉 BP | 🧪 Liver <br><br>
    📊 Accurate + Fast + User Friendly
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🚀 Start Health Check"):
            st.session_state.page = "upload"
            st.rerun()


# ================== UPLOAD PAGE ==================
elif st.session_state.page == "upload":

    st.title("📄 Upload Medical Report")
    st.markdown("### Upload PDF or Image Report (Max 5 Files)")

    uploaded_files = st.file_uploader(
        "Upload Reports",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) > 5:
            st.error("❌ Maximum 5 files allowed. Upload only 5 reports.")
            st.stop()

        full_text = ""

        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_image(file)

            full_text += "\n" + text

        st.session_state.report_text = full_text
        st.success("✅ OCR Completed Successfully!")

        if st.button("🔍 Predict Now"):

            lung_model, dia_model, heart_model, bp_model, liver_model = load_models()
            text = st.session_state.report_text

            diseases = detect_diseases(text)

            if len(diseases) == 0:
                st.error("❌ Disease not detected. Please upload valid report.")
                st.stop()

            age = int(get_value(r"Age\s*[:\-]?\s*(\d+)", text, 30))
            bmi = get_value(r"BMI\s*[:\-]?\s*(\d+\.?\d*)", text, 22)

            g = 1
            results = {}

            if "bp" in diseases:
                salt = get_value(r"Salt\s*[:\-]?\s*(\d+\.?\d*)", text, 10)
                bp_hist = 1
                smoking = 1
                results["bp"] = bp_model.predict_proba([[age, bmi, salt, bp_hist, smoking]])[0][1]

            if "dia" in diseases:
                hypertension = 1
                heart_disease = 1
                results["dia"] = dia_model.predict_proba([[g, age, hypertension, heart_disease, bmi]])[0][1]

            if "heart" in diseases:
                chol = get_value(r"Cholesterol\s*[:\-]?\s*(\d+\.?\d*)", text, 200)
                max_hr = int(get_value(r"Heart Rate\s*[:\-]?\s*(\d+)", text, 120))
                thal = 2
                results["heart"] = heart_model.predict_proba([[age, g, chol, max_hr, thal]])[0][1]

            if "liver" in diseases:
                tot_bil = get_value(r"Total Bilirubin\s*[:\-]?\s*(\d+\.?\d*)", text, 2.0)
                dir_bil = get_value(r"Direct Bilirubin\s*[:\-]?\s*(\d+\.?\d*)", text, 1.0)
                prot = get_value(r"Proteins\s*[:\-]?\s*(\d+\.?\d*)", text, 150)
                alb = get_value(r"Albumin\s*[:\-]?\s*(\d+\.?\d*)", text, 30)
                ratio = get_value(r"A/G Ratio\s*[:\-]?\s*(\d+\.?\d*)", text, 0.8)
                results["liver"] = liver_model.predict_proba([[age, g, tot_bil, dir_bil, prot, alb, ratio]])[0][1]

            if "lung" in diseases:
                smoking = 1
                alcohol = 1
                breath = 1
                chest = 1
                results["lung"] = lung_model.predict_proba([[g, age, smoking, alcohol, breath, chest]])[0][1]

            st.session_state.results = results
            st.session_state.page = "result"
            st.rerun()


# ================== RESULT PAGE ==================
elif st.session_state.page == "result":

    st.title("📊 Prediction Results")

    def risk_label(p):
        if p > 0.6:
            return "🔴 High Risk"
        elif p > 0.3:
            return "🟡 Moderate Risk"
        else:
            return "🟢 Low Risk"

    r = st.session_state.results

    if "bp" in r:
        st.write(f"💉 Hypertension Risk: {round(r['bp']*100,2)}% → {risk_label(r['bp'])}")
        st.progress(r['bp'])

    if "dia" in r:
        st.write(f"🩸 Diabetes Risk: {round(r['dia']*100,2)}% → {risk_label(r['dia'])}")
        st.progress(r['dia'])

    if "heart" in r:
        st.write(f"❤️ Heart Disease Risk: {round(r['heart']*100,2)}% → {risk_label(r['heart'])}")
        st.progress(r['heart'])

    if "lung" in r:
        st.write(f"🫁 Lung Cancer Risk: {round(r['lung']*100,2)}% → {risk_label(r['lung'])}")
        st.progress(r['lung'])

    if "liver" in r:
        st.write(f"🧪 Liver Disease Risk: {round(r['liver']*100,2)}% → {risk_label(r['liver'])}")
        st.progress(r['liver'])

    if st.button("➡️ Finish"):
        st.session_state.page = "thankyou"
        st.rerun()


# ================== THANK YOU PAGE ==================
elif st.session_state.page == "thankyou":

    st.title("🙏 Thank You for Visiting")

    st.markdown("""
    <div style='text-align:center; font-size:20px;'>
    💙 Stay Healthy <br><br>
    🧠 Your health matters!
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Restart"):
        st.session_state.page = "welcome"
        st.rerun()