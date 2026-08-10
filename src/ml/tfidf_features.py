# Import the libraries needed for this beginner-friendly text feature setup.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the path to the cleaned CSV using pathlib.
# This script reads the CSV and does not change it.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"
vectorizer_output_path = project_root / "models" / "tfidf_vectorizer.pkl"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned dataset into a DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare the target column and text columns
# ---------------------------------------------------
# The target column is fraudulent.
# The text columns will be combined into one text feature.
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Confirm that the target and text columns exist.
if "fraudulent" not in df.columns:
    raise ValueError("The 'fraudulent' column is required before splitting.")

for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not present in the dataset.")

# Store the target label as y.
y = df["fraudulent"]

# ---------------------------------------------------
# 4. Build a combined text field for each row
# ---------------------------------------------------
# Combine the five text columns into one text feature.
# Missing text is handled safely by converting missing values to empty strings.
# This happens only inside the analysis process and does not change the CSV.
df[text_columns] = df[text_columns].fillna("")

# Combine the chosen text columns into one text string per row.
# The blank values are skipped so that a missing field does not create stray spaces.
df["combined_text"] = df[text_columns].astype(str).agg(
    lambda row: " ".join(value.strip() for value in row if str(value).strip()),
    axis=1,
)

# ---------------------------------------------------
# 5. Apply the existing 80/20 stratified split
# ---------------------------------------------------
# This split is the same supervised-learning setup requested for the project.
# It keeps the target proportions similar in both subsets.
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
# This is a text-only feature setup.
# The vectorizer learns vocabulary from the training documents only.
vectorizer = TfidfVectorizer()

# Fit the vectorizer using only the training text.
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing text with the same fitted vectorizer.
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 7. Print the requested feature summary
# ---------------------------------------------------
print("TF-IDF feature extraction summary")
print("=" * 60)
print("Number of training documents:", X_train.shape[0])
print("Number of testing documents:", X_test.shape[0])
print("Number of TF-IDF features:", X_train_tfidf.shape[1])
print("Training TF-IDF matrix shape:", X_train_tfidf.shape)
print("Testing TF-IDF matrix shape:", X_test_tfidf.shape)
print()

# ---------------------------------------------------
# 8. Save the fitted vectorizer for later use
# ---------------------------------------------------
# Create the models directory if it does not exist.
vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the fitted vectorizer using joblib.
joblib.dump(vectorizer, vectorizer_output_path)

print(f"Fitted TF-IDF vectorizer saved to: {vectorizer_output_path}")
print("TF-IDF setup complete. The CSV files were not modified.")
