# Import the libraries this beginner-friendly threshold analysis needs.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# These paths point to the cleaned CSV and the saved model artifacts.
# This script reads the CSV and model files, and does not modify them.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"
model_path = project_root / "models" / "linear_svm_model.pkl"
vectorizer_path = project_root / "models" / "linear_svm_tfidf.pkl"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare the target and text columns
# ---------------------------------------------------
# The target column is the label we want to predict.
# The five text columns are the same ones used in the training scripts.
target_column = "fraudulent"
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Confirm the needed columns exist.
if target_column not in df.columns:
    raise ValueError("The 'fraudulent' column is required before analysis.")

for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not present in the dataset.")

# Store the target label in y.
y = df[target_column]

# ---------------------------------------------------
# 4. Build the combined text feature
# ---------------------------------------------------
# Fill missing text values with empty strings so the combined text is safe.
df[text_columns] = df[text_columns].fillna("")

# Combine the text fields into one text string for each row.
df["combined_text"] = df[text_columns].astype(str).agg(
    lambda row: " ".join(value.strip() for value in row if str(value).strip()),
    axis=1,
)

# ---------------------------------------------------
# 5. Recreate the same train-test split
# ---------------------------------------------------
# Recreate the 80/20 stratified train-test split with random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------------------------------------------
# 6. Load the saved vectorizer and model
# ---------------------------------------------------
# Load the fitted TF-IDF vectorizer created during the training workflow.
# This vectorizer should match the vocabulary that was learned on the training text.
vectorizer = joblib.load(vectorizer_path)

# Load the already-trained LinearSVC model.
# We do not retrain it in this script.
model = joblib.load(model_path)

# ---------------------------------------------------
# 7. Transform the test text using the saved vectorizer
# ---------------------------------------------------
# This transforms the testing documents with the same vocabulary learned earlier.
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 8. Obtain decision scores from the model
# ---------------------------------------------------
# LinearSVC uses decision_function() to create raw scores.
# These scores are not probabilities.
decision_scores = model.decision_function(X_test_tfidf)

# ---------------------------------------------------
# 9. Helper function for threshold evaluation
# ---------------------------------------------------
# A threshold is used to convert a score into a prediction.
# If a score is greater than or equal to the threshold, the prediction is 1.
# Otherwise, the prediction is 0.
def evaluate_threshold(scores, threshold):
    """Return metrics for a threshold on the decision scores."""

    # Convert the scores to binary predictions using the threshold.
    predictions = (scores >= threshold).astype(int)

    # Calculate the requested metrics.
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
    }

# ---------------------------------------------------
# 10. Analyze several thresholds
# ---------------------------------------------------
# Test the requested threshold values.
thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Store the threshold results in a list.
results = []

# Loop through the requested threshold values.
for threshold in thresholds:
    result = evaluate_threshold(decision_scores, threshold)
    results.append(result)

# ---------------------------------------------------
# 11. Print the threshold comparison table
# ---------------------------------------------------
print("LinearSVC threshold analysis using saved model and vectorizer")
print("=" * 80)
print("Threshold results for Fake Jobs (class 1)")
print()

# Build a friendly table with one row per threshold.
summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print()

# ---------------------------------------------------
# 12. Print the default LinearSVC threshold result at 0.0
# ---------------------------------------------------
# The default decision rule used by LinearSVC is threshold = 0.0.
# We already analyzed 0.0, but we print it clearly as requested.
default_predictions = (decision_scores >= 0.0).astype(int)
print("Default LinearSVC threshold result at 0.0")
print("=" * 80)
print("Accuracy:", accuracy_score(y_test, default_predictions))
print("Fake-job precision:", precision_score(y_test, default_predictions, pos_label=1, zero_division=0))
print("Fake-job recall:", recall_score(y_test, default_predictions, pos_label=1, zero_division=0))
print("Fake-job F1-score:", f1_score(y_test, default_predictions, pos_label=1, zero_division=0))
print()

# ---------------------------------------------------
# 13. End of script
# ---------------------------------------------------
print("Threshold analysis complete. The CSV file was not modified.")
