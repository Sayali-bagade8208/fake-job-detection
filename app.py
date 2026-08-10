# Import required Streamlit and project libraries.
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

from src.predict import predict_job

# ---------------------------------------------------
# 0. Paths for CSV prediction history storage
# ---------------------------------------------------
# Build a path to the project root and the data folder.
# The data folder is used only for storing prediction audit records.
project_root = Path(__file__).resolve().parent
data_dir = project_root / "data"
history_file = data_dir / "prediction_history.csv"

# Create the data directory if it does not exist already.
# This keeps the Streamlit app from failing before writing history.
data_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------
# Configure the basic Streamlit page details.
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------
# 2. Title and app description
# ---------------------------------------------------
# Show the page title and short explanation for beginner users.
st.title("🔍 Fake Job Detection System")
st.write("Machine Learning based system for detecting potentially fraudulent job postings.")

# ---------------------------------------------------
# 3. Manual job posting form
# ---------------------------------------------------
# This section collects all five text fields required by the saved model.
st.subheader("Manual Job Posting")

job_title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
job_description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

# ---------------------------------------------------
# 4. Prediction button and result display
# ---------------------------------------------------
# When the user presses the button, send all five fields to the prediction API.
# The function is reused from the saved model pipeline in src/predict.py.
if st.button("Predict Job"):
    prediction, decision_score = predict_job(
        job_title,
        company_profile,
        job_description,
        requirements,
        benefits,
    )

    # ---------------------------------------------------
    # 5. Save each successful manual prediction to CSV
    # ---------------------------------------------------
    # Build a record that matches the five input fields and the model output.
    # The timestamp is stored to make each prediction easier to review later.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prediction_record = {
        "timestamp": timestamp,
        "job_title": job_title,
        "company_profile": company_profile,
        "description": job_description,
        "requirements": requirements,
        "benefits": benefits,
        "prediction": prediction,
        "decision_score": decision_score,
    }

    # Create the CSV file with the expected headers only if the file does not exist.
    # If it does exist already, append the new record without deleting anything.
    expected_columns = [
        "timestamp",
        "job_title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "prediction",
        "decision_score",
    ]

    if history_file.exists():
        history_df = pd.read_csv(history_file)
        history_df = pd.concat(
            [history_df, pd.DataFrame([prediction_record], columns=expected_columns)],
            ignore_index=True,
        )
    else:
        history_df = pd.DataFrame(columns=expected_columns)
        history_df = pd.concat(
            [history_df, pd.DataFrame([prediction_record], columns=expected_columns)],
            ignore_index=True,
        )

    history_df.to_csv(history_file, index=False)

    # Show the result in a clear layout.
    st.subheader("Prediction Result")

    if prediction == "Fake Job":
        st.error(f"⚠️ Prediction: {prediction}")
    else:
        st.success(f"✅ Prediction: {prediction}")

    # Show the decision score in a separate area.
    st.info(f"Decision Score: {decision_score}")
