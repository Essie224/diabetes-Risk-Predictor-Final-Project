import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image
import joblib


# Set page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Load model
model = joblib.load('model.pkl')

# Create storage directory and file if it doesn't exist
LOG_FILE = 'data/predictions.csv'
os.makedirs("data", exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
        'bmi', 'HbA1c_level', 'blood_glucose_level', 'prediction'
    ]).to_csv(LOG_FILE, index=False)

# Helper: Log prediction
def log_prediction(input_data, prediction):
    input_data['prediction'] = prediction
    df = pd.read_csv(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([input_data])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# Helper: Make prediction
def make_prediction(data):
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)[0]
    log_prediction(data, int(prediction))
    return prediction

# App title
st.title("🩺 Type-2 Diabetes Risk Prediction App")

# Create tabs
tabs = st.tabs(["📝 Introduction", "📊 Data Exploration", "🤖 Prediction Model", "📁 Collected Data"])

# ----------------------------------------
# Tab 1: Introduction
with tabs[0]:
    st.header("Project Overview")
    st.markdown("""
    ### 📌 Problem Statement
    Type-2 Diabetes is a widespread chronic illness that can be managed or prevented with early risk detection.

    ### 🎯 Project Objective
    This project uses a machine learning model to assess your risk level based on common health indicators such as:
    - Age
    - Body Mass Index (BMI)
    - Blood Glucose
    - HbA1c
    - Hypertension and Heart Disease History
    - Smoking History

    ### 📚 Dataset
    The dataset comes from public sources and includes thousands of anonymized patient records.
    """)

    try:
        img = Image.open("assets/images/diabetes_info.png")
        st.image(img, caption="Type-2 Diabetes Risk Factors", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image not found: assets/images/diabetes_info.png")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# ----------------------------------------
# Tab 2: Data Exploration
with tabs[1]:
    st.header("📊 Data Exploration")
    st.markdown("Here are some visualizations showing feature relationships with Diabetes:")

    for filename, caption in [
        ("heart_disease_vs_diabetes.png", "Heart Disease vs Diabetes"),
        ("gender_vs_diabetes.png", "Gender vs Diabetes"),
        ("smoking_vs_diabetes.png", "Smoking History vs Diabetes"),
        ("glucose_vs_diabetes.png", "Blood Glucose Level vs Diabetes (Binned)"),
        ("heatmap.png", "Feature Correlation Heatmap")
    ]:
        try:
            img = Image.open(f"assets/images/{filename}")
            st.subheader(f"📈 {caption}")
            st.image(img)
        except FileNotFoundError:
            st.warning(f"⚠️ Missing image: assets/images/{filename}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # Add classification report as an image
    st.markdown("## 📄 Classification Report (Model Evaluation)")
    try:
        img = Image.open("assets/images/classification_report.png")
        st.image(img, caption="Precision, Recall, F1-Score for Each Class", use_container_width=True)

        st.markdown("### 🧠 Interpretation of Results")
        st.markdown("""
        - **Precision**: Of all patients predicted to be diabetic, how many truly were.
        - **Recall**: Of all actual diabetic patients, how many were correctly predicted.
        - **F1-score**: A balance between precision and recall.
        - **Overall Accuracy**: 98% — indicating a strong performing model.

        🏆 The model generalizes well and is suitable for real-world risk detection tasks.
        """)
    except FileNotFoundError:
        st.warning("⚠️ Classification report image not found. Please ensure it's in `assets/images/`.")

# ----------------------------------------
# Tab 3: Prediction Model
with tabs[2]:
    st.header("🤖 Predict Your Risk of Type-2 Diabetes")

    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    gender_word = st.selectbox("Gender:", list(gender_map.keys()))
    gender = gender_map[gender_word]

    age = st.number_input("Age (years):", min_value=18, max_value=120)

    yes_no_map = {"Yes": 1, "No": 0}
    hypertension = st.selectbox("Do you have hypertension?", list(yes_no_map.keys()))
    heart_disease = st.selectbox("Do you have heart disease?", list(yes_no_map.keys()))
    hypertension = yes_no_map[hypertension]
    heart_disease = yes_no_map[heart_disease]

    labels = {
        0: "No Info", 1: "Current Smoker", 2: "Smoked",
        3: "Former Smoker", 4: "Never Smoked", 5: "Not a current smoker"
    }
    smoking_history = st.select_slider(
        "Smoking History:",
        options=list(labels.keys()),
        format_func=lambda x: labels[x]
    )

    bmi = st.number_input("Body Mass Index (BMI):", min_value=10.0, max_value=80.0)
    hba1c = st.number_input("HbA1c Level (%):", min_value=2.0, max_value=15.0)
    glucose = st.number_input("Blood Glucose Level (mg/dL):", min_value=40, max_value=500)

    if st.button("🧠 Predict Risk"):
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }
        pred = make_prediction(input_data)
        if pred == 1:
            st.error("⚠️ You are at HIGH risk of developing Type-2 Diabetes.")
        else:
            st.success("✅ You are at LOW risk of developing Type-2 Diabetes.")

# ----------------------------------------
# Tab 4: Collected Data
with tabs[3]:
    st.header("📁 Collected Prediction Data")
    try:
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="user_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error loading prediction log: {e}")
