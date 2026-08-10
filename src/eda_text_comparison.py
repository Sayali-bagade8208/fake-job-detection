# Import only the libraries requested for this beginner-friendly analysis.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Use pathlib to build the location of the cleaned CSV file.
# This script reads the data and does not change the CSV file.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
reports_dir = project_root / "reports"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV file into a Pandas DataFrame.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Choose the text columns to analyze
# ---------------------------------------------------
# These are the requested text columns.
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]

# Verify that the required columns are available.
for column in text_columns:
    if column not in df.columns:
        raise ValueError(f"The required text column '{column}' is not in the dataset.")

if "fraudulent" not in df.columns:
    raise ValueError("The 'fraudulent' column is required for the Real vs Fake comparison.")

# ---------------------------------------------------
# 4. Create a helper function for text-length statistics
# ---------------------------------------------------
# This function calculates the character length for each row.
# Missing or empty text is treated as a length of 0 for this analysis only.
def calculate_text_lengths_for_group(series):
    """Return a Series containing character-length values for one text column."""

    # Fill missing values with an empty string.
    # This is safe for analysis and does not modify the CSV.
    text_series = series.fillna("")

    # Convert to string and calculate the number of characters.
    # Empty strings become length 0.
    lengths = text_series.astype(str).str.len()

    return lengths

# ---------------------------------------------------
# 5. Print comparison statistics for each text column
# ---------------------------------------------------
# The main loop goes through each text column and prints the group stats.
for column in text_columns:
    # Calculate all row lengths for this column.
    all_lengths = calculate_text_lengths_for_group(df[column])

    # Create two subsets:
    # Real jobs are fraudulent = 0.
    # Fake jobs are fraudulent = 1.
    real_mask = df["fraudulent"] == 0
    fake_mask = df["fraudulent"] == 1

    real_lengths = calculate_text_lengths_for_group(df.loc[real_mask, column])
    fake_lengths = calculate_text_lengths_for_group(df.loc[fake_mask, column])

    print(f"Text length comparison for '{column}'")
    print("=" * 65)

    # Print the statistics for the Real jobs group.
    print("Real jobs (fraudulent = 0):")
    real_summary = pd.Series({
        "count": len(real_lengths),
        "mean": real_lengths.mean(),
        "median": real_lengths.median(),
        "min": real_lengths.min(),
        "max": real_lengths.max(),
    })
    print(real_summary.to_string())
    print()

    # Print the statistics for the Fake jobs group.
    print("Fake jobs (fraudulent = 1):")
    fake_summary = pd.Series({
        "count": len(fake_lengths),
        "mean": fake_lengths.mean(),
        "median": fake_lengths.median(),
        "min": fake_lengths.min(),
        "max": fake_lengths.max(),
    })
    print(fake_summary.to_string())
    print()

    # ---------------------------------------------------
    # 6. Create a box plot for this text column
    # ---------------------------------------------------
    # A box plot helps compare the distribution of text lengths.
    # One chart file is created for each text column.
    figure, ax = plt.subplots(figsize=(8, 6))

    # Use a list of group-length arrays.
    data_for_boxplot = [real_lengths, fake_lengths]

    # Draw the box plot.
    # The labels argument is not accepted by this Matplotlib version,
    # so we draw the boxes and then set the tick labels manually.
    ax.boxplot(data_for_boxplot, patch_artist=True)
    ax.set_xticklabels(["Real", "Fake"])

    # Add clear labels and title.
    ax.set_title(f"Text Length Comparison: {column}")
    ax.set_xlabel("Job Type")
    ax.set_ylabel("Text Length (characters)")

    # Improve visual layout.
    plt.tight_layout()

    # Create the reports directory if it does not exist.
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save the chart with the exact requested filename.
    output_chart = reports_dir / f"text_length_{column}.png"
    plt.savefig(output_chart)

    # Display the chart without blocking the terminal.
    plt.show(block=False)
    plt.close(figure)

    print(f"Saved box plot: {output_chart}")
    print("-" * 65)
    print()

# ---------------------------------------------------
# 7. Finish the script
# ---------------------------------------------------
# This script only reads the cleaned CSV and draws charts.
print("Text comparison analysis complete. The CSV file was not modified.")
