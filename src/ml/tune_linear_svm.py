# Import the libraries needed for this beginner-friendly tuning workflow.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the input CSV path and output artifact paths using pathlib.
# This script reads the cleaned CSV and writes model and report artifacts.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"
model_output_path = project_root / "models" / "final_linear_svm_model.pkl"
vectorizer_output_path = project_root / "models" / "final_tfidf_vectorizer.pkl"
tuning_results_output_path = project_root / "reports" / "linear_svm_tuning_results.csv"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV file into a DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare target and text information
# ---------------------------------------------------
# The target is the fake-job label.
# The five chosen text fields are combined into one text feature.
target_column = "fraudulent"
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Check that the target column exists.
if target_column not in df.columns:
    raise ValueError("The 'fraudulent' column is required before model training.")

# Check that each requested text column exists.
for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not present in the dataset.")

# Store the target variable.
y = df[target_column]

# ---------------------------------------------------
# 4. Build the combined text field
# ---------------------------------------------------
# Fill missing text values with empty strings for safe feature construction.
# This is analysis-only handling and does not alter the CSV.
df[text_columns] = df[text_columns].fillna("")

# Combine the five text columns into one string for each row.
df["combined_text"] = df[text_columns].astype(str).agg(
    lambda row: " ".join(value.strip() for value in row if str(value).strip()),
    axis=1,
)

# ---------------------------------------------------
# 5. Recreate the same train-test split
# ---------------------------------------------------
# This is the exact split setup used in the previous ML scripts.
# The test set remains untouched until final evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------------------------------------------
# 6. Fit the TF-IDF vectorizer only on training text
# ---------------------------------------------------
# Use only the training documents to learn the vocabulary.
# This is the correct way to avoid information leakage from the test set.
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 7. Create the GridSearchCV tuning grid
# ---------------------------------------------------
# Define the C values to tune.
param_grid = {
    "C": [0.1, 0.5, 1, 2, 5]
}

# Use 3-fold cross-validation.
# Keep the folds stratified so class proportions stay similar in folds.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Use Fake Job F1-score (class 1) as the main scoring target.
# This is a useful metric for imbalanced detection projects.
model = LinearSVC(class_weight="balanced", random_state=42)

# GridSearchCV runs the model with different C values.
# Because scoring is F1 for class 1, GridSearchCV keeps the best model for that metric.
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)

# Fit GridSearchCV on the training TF-IDF matrix and the training target.
# The test set is not used in tuning.
grid_search.fit(X_train_tfidf, y_train)

# ---------------------------------------------------
# 8. Print tuning results
# ---------------------------------------------------
# Print the best parameter value selected by the search.
print("LinearSVC Hyperparameter Tuning Results")
print("=" * 80)
print("Best C:", grid_search.best_params_["C"])
print("Best cross-validation Fake F1-score:", grid_search.best_score_)
print()

# Print the full results for all tested C values.
# The GridSearchCV cv_results_ dictionary contains the detailed metric values.
print("Results for all tested C values:")
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[["param_C", "mean_test_score", "std_test_score", "rank_test_score"]].to_string(index=False))
print()

# ---------------------------------------------------
# 9. Train the final tuned LinearSVC on the complete training set
# ---------------------------------------------------
# Take the best model from tuning and fit it again on all training rows.
# This uses the best C value found during validation.
best_c = grid_search.best_params_["C"]
final_model = LinearSVC(class_weight="balanced", C=best_c, max_iter=5000, random_state=42)
final_model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------
# 10. Evaluate the final model once on the untouched test set
# ---------------------------------------------------
# Use the test set only once for final evaluation.
y_pred = final_model.predict(X_test_tfidf)

# Accuracy is overall correct classification rate.
accuracy = accuracy_score(y_test, y_pred)

# Focus on class 1, the Fake Jobs class.
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

# Build a full classification report for the test set.
classification_report_text = classification_report(y_test, y_pred, target_names=["Real Jobs", "Fake Jobs"], digits=4)

# Generate a confusion matrix for the final prediction.
confusion = confusion_matrix(y_test, y_pred)

# ROC-AUC uses the decision scores from the LinearSVC model.
y_scores = final_model.decision_function(X_test_tfidf)
roc_auc = roc_auc_score(y_test, y_scores)

# ---------------------------------------------------
# 11. Print final evaluation on the test set
# ---------------------------------------------------
print("Final tuned model evaluation on the untouched test set")
print("=" * 80)
print("Accuracy:", accuracy)
print("Fake Precision:", precision)
print("Fake Recall:", recall)
print("Fake F1-score:", f1)
print()
print("Classification report:")
print(classification_report_text)
print()
print("Confusion matrix:")
print(confusion)
print()
print("ROC-AUC score:", roc_auc)
print()

# ---------------------------------------------------
# 12. Save the final artifacts and tuning evidence
# ---------------------------------------------------
# Create the models directory if it does not exist.
model_output_path.parent.mkdir(parents=True, exist_ok=True)
vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)
tuning_results_output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the final model using joblib.
joblib.dump(final_model, model_output_path)

# Save the fitted vectorizer used by the final model.
joblib.dump(vectorizer, vectorizer_output_path)

# Save the GridSearchCV result table to CSV.
cv_results.to_csv(tuning_results_output_path, index=False)

print(f"Saved final LinearSVC model to: {model_output_path}")
print(f"Saved final TF-IDF vectorizer to: {vectorizer_output_path}")
print(f"Saved tuning results to: {tuning_results_output_path}")
print("Hyperparameter tuning and final evaluation complete. The CSV files were not modified.")
