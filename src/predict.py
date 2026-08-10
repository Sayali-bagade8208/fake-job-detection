# Import the libraries needed for this reusable prediction module.
import pandas as pd
from pathlib import Path
import joblib

# ---------------------------------------------------
# 1. Define paths to the saved model artifacts
# ---------------------------------------------------
# Build the project root from the current file location.
# The module is designed to be imported from the project and used by Streamlit later.
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "final_linear_svm_model.pkl"
vectorizer_path = project_root / "models" / "final_tfidf_vectorizer.pkl"

# ---------------------------------------------------
# 2. Load the saved model and TF-IDF vectorizer
# ---------------------------------------------------
# Load the trained final LinearSVC model.
# Load the final fitted TF-IDF vectorizer that was learned during tuning.
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ---------------------------------------------------
# 3. Define the reusable prediction function
# ---------------------------------------------------
# This function accepts the five text fields from a new job posting.
# It returns the final prediction label and the raw decision score.
def predict_job(title, company_profile, description, requirements, benefits):
    """Predict whether a new job posting is Real or Fake using the saved model."""

    # Handle missing values safely.
    # If any argument is None, convert it to an empty string.
    title = "" if title is None else title
    company_profile = "" if company_profile is None else company_profile
    description = "" if description is None else description
    requirements = "" if requirements is None else requirements
    benefits = "" if benefits is None else benefits

    # Combine the five text fields in the same order used during training.
    combined_text = " ".join([
        str(title).strip(),
        str(company_profile).strip(),
        str(description).strip(),
        str(requirements).strip(),
        str(benefits).strip(),
    ])

    # Transform the new posting text using the saved TF-IDF vectorizer.
    # This vectorizer was already fitted on training text only.
    input_tfidf = vectorizer.transform([combined_text])

    # Use the saved LinearSVC model's decision_function() method.
    # This gives a raw decision score for the sample.
    decision_score = model.decision_function(input_tfidf)[0]

    # Apply the saved model's final threshold of 0.0.
    # A score greater than or equal to 0.0 means class 1 (Fake Job).
    # Otherwise it means class 0 (Real Job).
    prediction = 1 if decision_score >= 0.0 else 0

    # Convert the numeric prediction to a label that is beginner-friendly.
    predicted_label = "Fake Job" if prediction == 1 else "Real Job"

    return predicted_label, decision_score

# ---------------------------------------------------
# 4. Small test section for manual checking
# ---------------------------------------------------
if __name__ == "__main__":
    # Test 1 - Real-looking job
    real_title = "Software Engineer"
    real_company_profile = "Established technology company developing enterprise software products and cloud solutions."
    real_description = "We are looking for a Software Engineer to join our development team. The candidate will work with senior engineers to design, develop, test and maintain software applications."
    real_requirements = "Bachelor degree in Computer Science or related field. Knowledge of Python or Java, SQL, Git and software development practices. Good communication and problem solving skills."
    real_benefits = "Competitive salary, health insurance, paid leave, professional development opportunities and flexible working arrangements."

    # Call the prediction function for the first test.
    prediction_1, decision_score_1 = predict_job(
        real_title,
        real_company_profile,
        real_description,
        real_requirements,
        real_benefits,
    )

    # Print the result for the first test.
    print("Test 1 - Real-looking Job")
    print("Prediction:", prediction_1)
    print("Decision score:", decision_score_1)
    print()

    # Test 2 - Fake-looking job
    fake_title = "Work From Home Data Entry - Earn $5000 Weekly"
    fake_company_profile = "We provide amazing online earning opportunities and guaranteed income from home."
    fake_description = "No experience required. Start immediately and earn thousands of dollars every week. Limited positions available."
    fake_requirements = "No experience needed. Anyone can apply."
    fake_benefits = "Very high income, work from anywhere, instant joining and guaranteed earnings."

    # Call the prediction function for the second test.
    prediction_2, decision_score_2 = predict_job(
        fake_title,
        fake_company_profile,
        fake_description,
        fake_requirements,
        fake_benefits,
    )

    # Print the result for the second test.
    print("Test 2 - Fake-looking Job")
    print("Prediction:", prediction_2)
    print("Decision score:", decision_score_2)
