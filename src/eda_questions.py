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
output_chart_path = project_root / "reports" / "questions_vs_fraud.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Create a cross-tabulation
# ---------------------------------------------------
# Build a table that compares application-question usage with fraud labels.
has_questions_vs_fraud = pd.crosstab(df["has_questions"], df["fraudulent"])

# Print the cross-tabulation clearly.
print("Cross-tabulation: has_questions vs fraudulent")
print(has_questions_vs_fraud)
print()

# ---------------------------------------------------
# 4. Calculate percentages within each has_questions category
# ---------------------------------------------------
# Convert counts into percentages across each row.
has_questions_pct = has_questions_vs_fraud.div(has_questions_vs_fraud.sum(axis=1), axis=0) * 100

# Rename the columns to make the class names easier to read.
has_questions_pct = has_questions_pct.rename(columns={0: "Real Jobs (0)", 1: "Fake Jobs (1)"})

# Print the percentage table.
print("Percentage of Real and Fake jobs within each has_questions category")
print(has_questions_pct)
print()

# ---------------------------------------------------
# 5. Create a grouped bar chart
# ---------------------------------------------------
# Define the two categories to show.
question_labels = ["No Application Questions", "Has Application Questions"]

# Pull out the percentages for the two classes.
real_percent = has_questions_pct["Real Jobs (0)"].values
fake_percent = has_questions_pct["Fake Jobs (1)"].values

# Create a grouped bar chart.
bar_width = 0.35
x = range(len(question_labels))

plt.figure(figsize=(8, 6))
plt.bar(x, real_percent, width=bar_width, label="Real Jobs (0)", color="#2ca02c")
plt.bar([pos + bar_width for pos in x], fake_percent, width=bar_width, label="Fake Jobs (1)", color="#d62728")

# Add labels, title and legends.
plt.xticks([pos + bar_width / 2 for pos in x], question_labels)
plt.xlabel("Application Questions Status")
plt.ylabel("Percentage of Jobs (%)")
plt.title("Has Questions vs Fraudulent Job Percentage")
plt.legend()
plt.tight_layout()

# Save the chart image.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
