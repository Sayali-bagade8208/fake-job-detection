# Import the libraries needed for analysis and visualisation.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the file paths with pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "telecommuting_vs_fraud.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Create a cross-tabulation
# ---------------------------------------------------
# Build a table that compares telecommuting values with fraudulent labels.
telecommuting_vs_fraud = pd.crosstab(df["telecommuting"], df["fraudulent"])

# Print the cross-tabulation clearly.
print("Cross-tabulation: telecommuting vs fraudulent")
print(telecommuting_vs_fraud)
print()

# ---------------------------------------------------
# 4. Calculate percentages inside each telecommuting category
# ---------------------------------------------------
# Convert counts to row percentages so each telecommuting group totals 100%.
telecommuting_pct = telecommuting_vs_fraud.div(telecommuting_vs_fraud.sum(axis=1), axis=0) * 100

# Give the columns friendlier names.
telecommuting_pct = telecommuting_pct.rename(columns={0: "Real Jobs (0)", 1: "Fake Jobs (1)"})

# Print the percentage table.
print("Percentage of Real and Fake jobs within each telecommuting category")
print(telecommuting_pct)
print()

# ---------------------------------------------------
# 5. Create a grouped bar chart
# ---------------------------------------------------
# Define the category labels.
telecommuting_labels = ["No Telecommuting", "Telecommuting"]

# Pull out the percentage values for each category.
real_percent = telecommuting_pct["Real Jobs (0)"]
fake_percent = telecommuting_pct["Fake Jobs (1)"]

# Use a grouped bar chart.
bar_width = 0.35
x = range(len(telecommuting_labels))

plt.figure(figsize=(8, 6))
plt.bar(x, real_percent, width=bar_width, label="Real Jobs (0)", color="#2ca02c")
plt.bar([pos + bar_width for pos in x], fake_percent, width=bar_width, label="Fake Jobs (1)", color="#d62728")

# Make the chart readable.
plt.xticks([pos + bar_width / 2 for pos in x], telecommuting_labels)
plt.xlabel("Telecommuting Status")
plt.ylabel("Percentage of Jobs (%)")
plt.title("Telecommuting vs Fraudulent Job Percentage")
plt.legend()
plt.tight_layout()

# Save the chart image.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
