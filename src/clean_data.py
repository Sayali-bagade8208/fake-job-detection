# Import Pandas and pathlib.
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# 1. Define the file paths
# ---------------------------------------------------
# Use pathlib to build the input and output paths.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "fake_job_postings.csv"
output_path = project_root / "data" / "cleaned_job_postings.csv"

# ---------------------------------------------------
# 2. Load the original dataset
# ---------------------------------------------------
# Read the original CSV file into a DataFrame.
# This file must stay unchanged.
df = pd.read_csv(input_path)

# Save the before-cleaning row count and missing-value values.
rows_before = len(df)
missing_before = df.isna().sum()

# ---------------------------------------------------
# 3. Remove duplicate rows if any exist
# ---------------------------------------------------
# Drop duplicate rows and keep the first occurrence.
df = df.drop_duplicates()

# ---------------------------------------------------
# 4. Handle missing values for text/categorical columns
# ---------------------------------------------------
# Find text-like columns from the dataset.
# These are usually object/string columns.
text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

# Fill missing values in text/categorical columns with an empty string.
# This keeps the data ready for a later profile/check without dropping rows.
# Do not fill the target column fraudulent.
for column in text_columns:
    if column != "fraudulent":
        df[column] = df[column].fillna("")

# ---------------------------------------------------
# 5. Print cleaning summary
# ---------------------------------------------------
# Print row count before and after removing duplicates.
print("Rows before cleaning:", rows_before)
print("Rows after cleaning:", len(df))
print()

# Print missing values before and after cleaning.
print("Missing values before cleaning:")
print(missing_before)
print()

print("Missing values after cleaning:")
print(df.isna().sum())
print()

# ---------------------------------------------------
# 6. Save the cleaned dataframe to a new CSV file
# ---------------------------------------------------
# Keep the original 18 columns in the output.
# Save a separate copy without changing the original CSV.
df.to_csv(output_path, index=False)

print("Cleaned data saved to:", output_path)
