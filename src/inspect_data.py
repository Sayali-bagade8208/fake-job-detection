# Import the Pandas library for working with CSV data.
import pandas as pd
from pathlib import Path

# Build the path to the dataset from the project root.
# __file__ points to this script's location inside src/.
project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "data" / "fake_job_postings.csv"

# Load the CSV file into a DataFrame.
# A DataFrame is a table-like structure used for data analysis.
df = pd.read_csv(dataset_path)

# Display basic dataset shape.
print("1. Number of rows and columns")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print()

# Show the column names.
print("2. Column names")
print(df.columns.tolist())
print()

# Show the first 5 rows of the dataset.
print("3. First 5 rows")
print(df.head())
print()

# Show the data types for every column.
print("4. Data types of all columns")
print(df.dtypes)
print()

# Count missing values in each column.
print("5. Number of missing values in each column")
print(df.isnull().sum())
print()

# Count duplicate rows in the entire dataset.
print("6. Number of duplicate rows")
print("Duplicate rows:", df.duplicated().sum())
print()

# Show the distribution of the fraudulent column.
print("7. Value counts of the 'fraudulent' column")
print(df["fraudulent"].value_counts())
