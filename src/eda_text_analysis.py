# Import only the libraries requested for this beginner-friendly EDA script.
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the path to the cleaned dataset using pathlib.
# The script reads this file and does not change it.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the CSV file into a Pandas DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Choose the text columns to analyze
# ---------------------------------------------------
# These are the requested text fields.
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# ---------------------------------------------------
# 4. Helper function to calculate text statistics
# ---------------------------------------------------
# This function will be used for each text column.
# It collects the summary metrics for the dataset as a whole.
def summarize_text_column(df, column_name):
    """Return a one-row summary table for one text column."""

    # Total number of rows in the dataset.
    total_rows = len(df)

    # Identify rows where the value is present and not only whitespace.
    # Missing values are treated as not usable text.
    non_missing_and_non_empty = df[column_name].notna() & df[column_name].astype(str).str.strip().ne("")

    # Count valid text rows.
    valid_text_count = int(non_missing_and_non_empty.sum())

    # Count rows that are missing or empty after trimming spaces.
    missing_or_empty_count = total_rows - valid_text_count

    # Extract only the usable string values and measure their character lengths.
    text_lengths = df.loc[non_missing_and_non_empty, column_name].astype(str).str.len()

    # Guard against a column with no usable text values.
    if len(text_lengths) > 0:
        average_length = float(text_lengths.mean())
        minimum_length = int(text_lengths.min())
        maximum_length = int(text_lengths.max())
    else:
        average_length = 0.0
        minimum_length = 0
        maximum_length = 0

    # Return a small summary dictionary.
    return {
        "text_column": column_name,
        "total_rows": total_rows,
        "non_missing_non_empty": valid_text_count,
        "missing_or_empty": missing_or_empty_count,
        "average_text_length": average_length,
        "min_text_length": minimum_length,
        "max_text_length": maximum_length,
    }

# ---------------------------------------------------
# 5. Build the summary table for all requested text columns
# ---------------------------------------------------
# Create an empty list to hold the results for each text column.
summary_rows = []

# Loop through each requested text column.
# This keeps the script easy to follow for beginners.
for column in text_columns:
    # Make sure the column exists in the dataset.
    # If it does not exist, this script will stop with a clear error.
    if column not in df.columns:
        raise ValueError(f"The column '{column}' was not found in the dataset.")

    # Calculate the text statistics for one column.
    result = summarize_text_column(df, column)

    # Store the result in the list.
    summary_rows.append(result)

# Convert the calculated rows into a DataFrame.
summary_table = pd.DataFrame(summary_rows)

# ---------------------------------------------------
# 6. Print the full summary table
# ---------------------------------------------------
# Print a clean table for the text-column analysis.
print("Text column summary for the cleaned dataset")
print("=" * 55)
print(summary_table.to_string(index=False))
print()

# ---------------------------------------------------
# 7. Compare average text length by fraud label
# ---------------------------------------------------
# This part focuses only on average text length,
# not on any machine learning or preprocessing tasks.
print("Average text length comparison by fraudulent label")
print("=" * 55)

# Make a small table to display the real-job and fake-job averages.
comparison_rows = []

# Create masks for the two label values.
real_jobs_mask = df["fraudulent"] == 0
fake_jobs_mask = df["fraudulent"] == 1

# Compare each text column separately.
for column in text_columns:
    # Confirm that the target column exists for comparison.
    if column not in df.columns:
        raise ValueError(f"The column '{column}' was not found in the dataset.")

    # Use only non-empty and non-missing text values for length calculation.
    real_non_empty = real_jobs_mask & df[column].notna() & df[column].astype(str).str.strip().ne("")
    fake_non_empty = fake_jobs_mask & df[column].notna() & df[column].astype(str).str.strip().ne("")

    real_lengths = df.loc[real_non_empty, column].astype(str).str.len()
    fake_lengths = df.loc[fake_non_empty, column].astype(str).str.len()

    # Calculate the average length for the two groups.
    real_average = float(real_lengths.mean()) if len(real_lengths) > 0 else 0.0
    fake_average = float(fake_lengths.mean()) if len(fake_lengths) > 0 else 0.0

    # Add one row to the comparison table.
    comparison_rows.append({
        "text_column": column,
        "real_jobs_average_length": real_average,
        "fake_jobs_average_length": fake_average,
    })

# Display the comparison table.
comparison_table = pd.DataFrame(comparison_rows)
print(comparison_table.to_string(index=False))
print()

# ---------------------------------------------------
# 8. End of the script
# ---------------------------------------------------
# The dataset has been read-only and unchanged.
print("Text analysis complete. The source CSV was not modified.")
