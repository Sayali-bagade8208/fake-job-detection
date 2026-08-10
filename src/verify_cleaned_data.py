# Import Pandas and pathlib.
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# 1. Define the path to the cleaned dataset
# ---------------------------------------------------
# Use pathlib to build the file path.
project_root = Path(__file__).resolve().parents[1]
cleaned_path = project_root / "data" / "cleaned_job_postings.csv"

# ---------------------------------------------------
# 2. Load the cleaned CSV into a DataFrame
# ---------------------------------------------------
# Read the cleaned dataset.
df = pd.read_csv(cleaned_path)

# ---------------------------------------------------
# 3. Check basic dataset structure
# ---------------------------------------------------
print("1. Number of rows")
print(len(df))
print()

print("2. Number of columns")
print(len(df.columns))
print()

print("3. Column names")
print(list(df.columns))
print()

# ---------------------------------------------------
# 4. Check missing values
# ---------------------------------------------------
print("4. Missing values for every column")
print(df.isna().sum())
print()

# ---------------------------------------------------
# 5. Check duplicate rows
# ---------------------------------------------------
print("5. Number of duplicate rows")
print(df.duplicated().sum())
print()

# ---------------------------------------------------
# 6. Check target column distribution
# ---------------------------------------------------
print("6. Value counts of the fraudulent column")
print(df["fraudulent"].value_counts())
print()

# ---------------------------------------------------
# 7. Preview the cleaned dataset
# ---------------------------------------------------
print("7. First 5 rows")
print(df.head())
print()

# ---------------------------------------------------
# 8. Verify expected column count
# ---------------------------------------------------
expected_columns = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent"
]

if list(df.columns) == expected_columns:
    print("Verification: cleaned dataset contains the expected 18 columns.")
else:
    print("Verification: cleaned dataset does not contain the expected 18 columns.")
    print("Expected columns:", expected_columns)
    print("Actual columns:", list(df.columns))
