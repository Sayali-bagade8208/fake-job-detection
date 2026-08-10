# Import the Pandas library.
import pandas as pd
from pathlib import Path

# Build the dataset path.
project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "data" / "fake_job_postings.csv"

# Load the CSV file into a Pandas DataFrame.
df = pd.read_csv(dataset_path)

# Turn off Pandas display truncation so the summary table stays visible.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.expand_frame_repr", False)

# ---------------------------
# 1. Show the first 5 rows
# ---------------------------
print("First 5 rows of the dataset:")
print(df.head(5).to_string(index=False))
print()

# ------------------------------------
# 2. Build a column profile summary
# ------------------------------------
print("Column profile summary:")

# Create a simple summary table for every column.
profile = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.values,
    "Non-Null Count": df.notna().sum().values,
    "Missing Value Count": df.isnull().sum().values,
    "Unique Value Count": df.nunique().values
})

# Keep the column names exactly as requested and show the whole table.
profile = profile[["Column Name", "Data Type", "Non-Null Count", "Missing Value Count", "Unique Value Count"]]
print(profile.to_string(index=False))
