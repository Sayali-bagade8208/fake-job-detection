# Import the libraries needed for this beginner-friendly training script.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the input CSV path and model output paths using pathlib.
# This script reads the cleaned CSV and writes the model artifacts.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"
model_output_path = project_root / "models" / "logistic_regression_model.pkl"
vectorizer_output_path = project_root / "models" / "logistic_regression_tfidf.pkl"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV file into a DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare the target and text columns
# ---------------------------------------------------
# The target column is fraudulent.
# The five text columns are combined into one text feature for vectorization.
target_column = "fraudulent"
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Confirm the columns exist before using them.
if target_column not in df.columns:
    raise ValueError("The 'fraudulent' column is required before model training.")

for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not present in the dataset.")

# Store the target label in y.
y = df[target_column]

# ---------------------------------------------------
# 4. Build the combined text feature
# ---------------------------------------------------
# Fill missing text values safely with empty strings.
# This is only for the analysis/training flow and does not update the CSV.
df[text_columns] = df[text_columns].fillna("")

# Combine the selected text columns into a single text string per row.
# This is the same logic used in the TF-IDF feature preparation script.
df["combined_text"] = df[text_columns].astype(str).agg(
    lambda row: " ".join(value.strip() for value in row if str(value).strip()),
    axis=1,
)

# ---------------------------------------------------
# 5. Create the stratified train-test split
# ---------------------------------------------------
# Use the same split setup as before: 80% training and 20% testing.
# Stratification keeps the class proportions similar across both sets.
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
# The vectorizer learns vocabulary using only the training documents.
# The testing documents are transformed using the fitted vocabulary.
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 7. Train the logistic regression model
# ---------------------------------------------------
# Use class_weight='balanced' to give more attention to the minority fake-job class.
# max_iter=1000 gives the solver enough iterations to converge.
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

# Fit the model only on the training TF-IDF features and training target.
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------
# 8. Generate predictions and evaluate the model
# ---------------------------------------------------
# Use the fitted model to predict the fake-job label for the testing set.
y_pred = model.predict(X_test_tfidf)

# Accuracy is the overall proportion of correct predictions.
accuracy = accuracy_score(y_test, y_pred)

# For class 1 (fake jobs), it is useful to look at class-1 precision, recall, and F1.
# The average='binary' setting measures the positive class, which is label 1 here.
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

# Get a full classification report for the 0/1 labels.
classification_report_text = classification_report(y_test, y_pred, target_names=["Real Jobs", "Fake Jobs"], digits=4)

# Build the confusion matrix.
confusion = confusion_matrix(y_test, y_pred)

# Compute the ROC-AUC score using the positive class probabilities.
# For binary classification, the probability of class 1 is the important signal.
y_test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_probabilities)

# ---------------------------------------------------
# 9. Print model evaluation results
# ---------------------------------------------------
print("Logistic Regression Evaluation")
print("=" * 80)
print("Accuracy:", accuracy)
print("Precision (class 1 / Fake Jobs):", precision)
print("Recall (class 1 / Fake Jobs):", recall)
print("F1-score (class 1 / Fake Jobs):", f1)
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
# 10. Save the fitted model and vectorizer objects
# ---------------------------------------------------
# Create the models directory if it does not already exist.
model_output_path.parent.mkdir(parents=True, exist_ok=True)
vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the fitted logistic regression model to disk.
joblib.dump(model, model_output_path)

# Save the fitted TF-IDF vectorizer used for the same features.
joblib.dump(vectorizer, vectorizer_output_path)

print(f"Saved trained model to: {model_output_path}")
print(f"Saved fitted TF-IDF vectorizer to: {vectorizer_output_path}")
print("Training complete. The CSV files were not modified.")
