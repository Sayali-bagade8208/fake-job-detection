# Import the libraries needed for analysis and charting.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build paths with pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "employment_type_vs_fraud.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Show unique employment_type values
# ---------------------------------------------------
# Print the categories that appear in the dataset.
print("Unique employment_type values:")
print(df["employment_type"].dropna().unique())
print()

# ---------------------------------------------------
# 4. Handle missing/empty employment_type values safely
# ---------------------------------------------------
# Replace blank strings and null values with "Unknown" for display purposes.
# This is only for the EDA chart, not for changing the dataset file.
df["employment_type"] = df["employment_type"].fillna("Unknown")
df["employment_type"] = df["employment_type"].replace("", "Unknown")

# ---------------------------------------------------
# 5. Create a cross-tabulation
# ---------------------------------------------------
# Build a table showing employment type counts for each fraud label.
employment_vs_fraud = pd.crosstab(df["employment_type"], df["fraudulent"])

# Print the count table clearly.
print("Cross-tabulation: employment_type vs fraudulent")
print(employment_vs_fraud)
print()

# ---------------------------------------------------
# 6. Calculate percentages within each employment type
# ---------------------------------------------------
# Convert each row to percentages so the categories sum to 100%.
employment_pct = employment_vs_fraud.div(employment_vs_fraud.sum(axis=1), axis=0) * 100

# Rename the columns to make them easy to understand.
employment_pct = employment_pct.rename(columns={0: "Real Jobs (0)", 1: "Fake Jobs (1)"})

# Print the percentage table clearly.
print("Percentage of Real and Fake jobs within each employment type")
print(employment_pct)
print()

# ---------------------------------------------------
# 7. Create a grouped bar chart for percentages
# ---------------------------------------------------
# Extract labels and percentages.
employment_labels = list(employment_pct.index)
real_percent = employment_pct["Real Jobs (0)"].values
fake_percent = employment_pct["Fake Jobs (1)"].values

# Create a grouped bar chart.
bar_width = 0.35
x = range(len(employment_labels))

plt.figure(figsize=(10, 6))
plt.bar(x, real_percent, width=bar_width, label="Real Jobs (0)", color="#2ca02c")
plt.bar([pos + bar_width for pos in x], fake_percent, width=bar_width, label="Fake Jobs (1)", color="#d62728")

# Add axis labels, title, legend, and x-ticks.
plt.xticks([pos + bar_width / 2 for pos in x], employment_labels, rotation=45, ha="right")
plt.xlabel("Employment Type")
plt.ylabel("Percentage of Jobs (%)")
plt.title("Employment Type vs Fraudulent Job Percentage")
plt.legend()
plt.tight_layout()

# Save the chart image.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
