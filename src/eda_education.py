# Import the libraries needed for analysis and charting.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build paths using pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "education_vs_fraud.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Show unique required_education values
# ---------------------------------------------------
# Print the categories that appear in the dataset.
print("Unique required_education values:")
print(df["required_education"].dropna().unique())
print()

# ---------------------------------------------------
# 4. Handle missing/empty education values safely
# ---------------------------------------------------
# Replace blank strings and null values with "Unknown" for display purposes.
# This does not change the saved CSV file.
df["required_education"] = df["required_education"].fillna("Unknown")
df["required_education"] = df["required_education"].replace("", "Unknown")

# ---------------------------------------------------
# 5. Create a cross-tabulation
# ---------------------------------------------------
# Build a count table for education vs fraudulent.
education_vs_fraud = pd.crosstab(df["required_education"], df["fraudulent"])

# Print the count table clearly.
print("Cross-tabulation: required_education vs fraudulent")
print(education_vs_fraud)
print()

# ---------------------------------------------------
# 6. Calculate percentages within each education category
# ---------------------------------------------------
# Convert counts into percentages for each education category.
education_pct = education_vs_fraud.div(education_vs_fraud.sum(axis=1), axis=0) * 100

# Rename the columns to make them easier to read.
education_pct = education_pct.rename(columns={0: "Real Jobs (0)", 1: "Fake Jobs (1)"})

# Print the percentage table clearly.
print("Percentage of Real and Fake jobs within each required_education category")
print(education_pct)
print()

# ---------------------------------------------------
# 7. Create a grouped bar chart for percentages
# ---------------------------------------------------
# Extract labels and percentage values.
education_labels = list(education_pct.index)
real_percent = education_pct["Real Jobs (0)"].values
fake_percent = education_pct["Fake Jobs (1)"].values

# Create a grouped bar chart.
bar_width = 0.35
x = range(len(education_labels))

plt.figure(figsize=(12, 6))
plt.bar(x, real_percent, width=bar_width, label="Real Jobs (0)", color="#2ca02c")
plt.bar([pos + bar_width for pos in x], fake_percent, width=bar_width, label="Fake Jobs (1)", color="#d62728")

# Add labels, title, legend, and x-ticks.
plt.xticks([pos + bar_width / 2 for pos in x], education_labels, rotation=45, ha="right")
plt.xlabel("Required Education")
plt.ylabel("Percentage of Jobs (%)")
plt.title("Required Education vs Fraudulent Job Percentage")
plt.legend()
plt.tight_layout()

# Save the chart image.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
