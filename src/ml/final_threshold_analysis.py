# Import the libraries needed for this beginner-friendly threshold analysis.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the paths to the cleaned CSV and the saved final model artifacts.
# This script reads the CSV and saved artifacts and does not modify them.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"
model_path = project_root / "models" / "final_linear_svm_model.pkl"
vectorizer_path = project_root / "models" / "final_tfidf_vectorizer.pkl"
reports_dir = project_root / "reports"
output_chart_path = reports_dir / "final_threshold_comparison.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned dataset from the CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare the target and text columns
# ---------------------------------------------------
# The project target label is fraudulent.
# The five text columns are the same ones used earlier for the text features.
target_column = "fraudulent"
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Confirm the required columns are available.
if target_column not in df.columns:
    raise ValueError("The 'fraudulent' column is required before analysis.")

for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not present in the dataset.")

# Save the target label for the later evaluation.
y = df[target_column]

# ---------------------------------------------------
# 4. Build the combined text feature
# ---------------------------------------------------
# Missing text is handled safely by replacing missing values with empty strings.
# This is analysis-only behavior and does not modify the CSV.
df[text_columns] = df[text_columns].fillna("")

# Combine the five text fields into one text string per row.
df["combined_text"] = df[text_columns].astype(str).agg(
    lambda row: " ".join(value.strip() for value in row if str(value).strip()),
    axis=1,
)

# ---------------------------------------------------
# 5. Recreate the same stratified train-test split
# ---------------------------------------------------
# Build the exact same valid 80/20 split with random_state=42.
# The split is used to keep the evaluation consistent with model training.
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------------------------------------------
# 6. Load the saved final TF-IDF vectorizer and model
# ---------------------------------------------------
# Load the fitted vocabulary from the final TF-IDF artifact.
vectorizer = joblib.load(vectorizer_path)

# Load the final tuned LinearSVC model object.
model = joblib.load(model_path)

# ---------------------------------------------------
# 7. Transform the test text using the saved vectorizer
# ---------------------------------------------------
# The test text is transformed using the same vocabulary from training.
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 8. Generate decision scores from the saved LinearSVC model
# ---------------------------------------------------
# LinearSVC prediction uses decision_function() scores rather than probabilities.
decision_scores = model.decision_function(X_test_tfidf)

# ---------------------------------------------------
# 9. Helper function to evaluate one threshold
# ---------------------------------------------------
# A threshold is applied to each decision score.
# If the decision score is greater than or equal to the threshold,
# we predict the Fake Job class as 1. Otherwise we predict 0.
def evaluate_threshold(scores, threshold):
    """Return fake-job metric values for a threshold."""

    # Convert scores to binary predictions.
    predictions = (scores >= threshold).astype(int)

    # Compute the requested metric values.
    accuracy = accuracy_score(y_test, predictions)
    fake_precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
    fake_recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    fake_f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
    }

# ---------------------------------------------------
# 10. Evaluate the requested thresholds
# ---------------------------------------------------
# These are the thresholds requested by the user.
threshold_values = [-0.50, -0.25, 0.00, 0.25, 0.50]

# Evaluate each requested threshold one by one.
results = []
for threshold in threshold_values:
    result = evaluate_threshold(decision_scores, threshold)
    results.append(result)

# Convert the list of results into a DataFrame for a clean printed table.
threshold_df = pd.DataFrame(results)

# ---------------------------------------------------
# 11. Print the threshold comparison table
# ---------------------------------------------------
print("Final tuned LinearSVC threshold analysis")
print("=" * 90)
print(threshold_df.to_string(index=False))
print()

# ---------------------------------------------------
# 12. Print the default threshold result at 0.0
# ---------------------------------------------------
# The LinearSVC default decision threshold in binary classification is 0.0.
default_predictions = (decision_scores >= 0.0).astype(int)

print("Default LinearSVC threshold result at 0.0")
print("=" * 90)
print("Accuracy:", accuracy_score(y_test, default_predictions))
print("Fake Precision:", precision_score(y_test, default_predictions, pos_label=1, zero_division=0))
print("Fake Recall:", recall_score(y_test, default_predictions, pos_label=1, zero_division=0))
print("Fake F1-score:", f1_score(y_test, default_predictions, pos_label=1, zero_division=0))
print()

# ---------------------------------------------------
# 13. Create the chart for the threshold comparison
# ---------------------------------------------------
# The threshold table is now ready to be shown in a chart.
# Use the threshold column as the x-axis and the metric values as plotted lines.
# The y-axis is the metric score value from 0 to 1.
figure, ax = plt.subplots(figsize=(10, 6))

# Plot the requested metrics as separate lines.
ax.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="Accuracy")
ax.plot(threshold_df["threshold"], threshold_df["fake_precision"], marker="o", label="Fake Precision")
ax.plot(threshold_df["threshold"], threshold_df["fake_recall"], marker="o", label="Fake Recall")
ax.plot(threshold_df["threshold"], threshold_df["fake_f1"], marker="o", label="Fake F1")

# Add chart labels and title.
ax.set_title("Final LinearSVC Decision Threshold Comparison")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Ensure the reports directory exists.
reports_dir.mkdir(parents=True, exist_ok=True)

# Save the chart to the requested file name.
plt.savefig(output_chart_path)

# Display the figure without blocking the terminal.
plt.show(block=False)
plt.close(figure)

print(f"Saved threshold comparison chart to: {output_chart_path}")
print("Threshold analysis complete. The CSV files were not modified.")
