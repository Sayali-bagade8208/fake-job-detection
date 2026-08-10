# Import the libraries needed for analysis and charting.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Create paths using pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "company_logo_vs_fraud.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Create a cross-tabulation
# ---------------------------------------------------
# Create a table that shows how often each logo flag appears with each label.
logo_vs_fraud = pd.crosstab(df["has_company_logo"], df["fraudulent"])

# Print the cross-tabulation clearly.
print("Cross-tabulation: has_company_logo vs fraudulent")
print(logo_vs_fraud)
print()

# ---------------------------------------------------
# 4. Calculate percentages inside each logo category
# ---------------------------------------------------
# Convert the cross-tab into row percentages so we see the distribution
# inside each logo category.
logo_percentage = logo_vs_fraud.div(logo_vs_fraud.sum(axis=1), axis=0) * 100

# Rename the columns to match the class labels for clarity.
logo_percentage = logo_percentage.rename(columns={0: "Real Jobs (0)", 1: "Fake Jobs (1)"})

# Print the percentage table.
print("Percentage of Real and Fake jobs within each company-logo category")
print(logo_percentage)
print()

# ---------------------------------------------------
# 5. Create a bar chart for the percentages
# ---------------------------------------------------
# Get the x-axis labels.
logo_labels = ["No Company Logo", "Company Logo"]

# Build the chart data.
# Each category has two values: real jobs and fake jobs percentages.
real_percent = logo_percentage["Real Jobs (0)"].values
fake_percent = logo_percentage["Fake Jobs (1)"].values

# Use a grouped bar chart.
bar_width = 0.35
x = range(len(logo_labels))

plt.figure(figsize=(8, 6))
plt.bar(x, real_percent, width=bar_width, label="Real Jobs (0)", color="#2ca02c")
plt.bar([pos + bar_width for pos in x], fake_percent, width=bar_width, label="Fake Jobs (1)", color="#d62728")

# Add useful labels and a title.
plt.xticks([pos + bar_width / 2 for pos in x], logo_labels)
plt.xlabel("Company Logo Status")
plt.ylabel("Percentage of Jobs (%)")
plt.title("Company Logo vs Fraudulent Job Percentage")
plt.legend()
plt.tight_layout()

# Save the chart.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
