# Import the libraries needed for this beginner-friendly split script.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the path to the cleaned CSV file using pathlib.
# This script reads the dataset and does not write to it.
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "cleaned_job_postings.csv"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Load the cleaned CSV file into a Pandas DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Prepare the target and feature data
# ---------------------------------------------------
# Separate the target column y from all remaining columns.
# The remaining columns are the features X.
y = df["fraudulent"]
X = df.drop(columns=["fraudulent"])

# ---------------------------------------------------
# 4. Create an 80/20 split with stratification
# ---------------------------------------------------
# Train_test_split splits the dataset into training and testing parts.
# - test_size=0.2 means 20% of the dataset goes to the test set.
# - random_state=42 keeps the split reproducible.
# - stratify=y keeps the target class proportions similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# 5. Print a simple split summary
# ---------------------------------------------------
# Print the dataset size and the split sizes.
print("Fake Job Detection: train-test split")
print("=" * 50)
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()

# Print class distribution for the training set.
print("Training set distribution for 'fraudulent':")
print(y_train.value_counts())
print()

# Print class distribution for the testing set.
print("Testing set distribution for 'fraudulent':")
print(y_test.value_counts())
print()

# ---------------------------------------------------
# 6. End of the script
# ---------------------------------------------------
# The CSV files remain unchanged.
print("Train/test split complete. The CSV files were not modified.")
