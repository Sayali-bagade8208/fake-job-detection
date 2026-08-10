# Import the libraries needed for analysis and charting.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the paths using pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "top_functions_by_fake_jobs.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Treat missing/empty function values safely
# ---------------------------------------------------
# Replace missing values and empty strings with "Unknown" for analysis only.
# Do not edit the CSV file.
df["function"] = df["function"].fillna("Unknown")
df["function"] = df["function"].replace("", "Unknown")

# ---------------------------------------------------
# 4. Calculate function-level counts and percentages
# ---------------------------------------------------
# Group by the function column and summarize the target label.
function_summary = df.groupby("function").agg(
    total_postings=("fraudulent", "size"),
    real_jobs=("fraudulent", lambda x: (x == 0).sum()),
    fake_jobs=("fraudulent", lambda x: (x == 1).sum())
)

# Add the fake-job percentage column.
function_summary["fake_job_percentage"] = (function_summary["fake_jobs"] / function_summary["total_postings"]) * 100

# Reset the index for a friendly printed table.
function_summary = function_summary.reset_index()

# Print the summary table.
print("Function-level summary:")
print(function_summary)
print()

# ---------------------------------------------------
# 5. Print top 15 functions by fake-job count
# ---------------------------------------------------
# Sort by the fake jobs count from high to low.
top_fake_count = function_summary.sort_values("fake_jobs", ascending=False).head(15)

print("Top 15 functions by number of fake job postings:")
print(top_fake_count[["function", "fake_jobs", "total_postings", "fake_job_percentage"]])
print()

# ---------------------------------------------------
# 6. Print top 15 functions by fake-job percentage
# ---------------------------------------------------
# Keep only categories with at least 30 total job postings.
filtered_for_percentage = function_summary[function_summary["total_postings"] >= 30]

# Sort by fake job percentage from high to low.
top_fake_percentage = filtered_for_percentage.sort_values("fake_job_percentage", ascending=False).head(15)

print("Top 15 functions by fake-job percentage (minimum 30 postings):")
print(top_fake_percentage[["function", "fake_jobs", "total_postings", "fake_job_percentage"]])
print()

# ---------------------------------------------------
# 7. Create a horizontal bar chart for the top fake-count functions
# ---------------------------------------------------
# Use the top 15 fake-count functions for plotting.
plot_data = top_fake_count.sort_values("fake_jobs", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(plot_data["function"], plot_data["fake_jobs"], color="#d62728")

# Add labels and title.
plt.xlabel("Number of Fake Job Postings")
plt.ylabel("Function")
plt.title("Top 15 Functions by Number of Fake Job Postings")
plt.tight_layout()

# Save the chart to the reports folder.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without blocking the terminal.
plt.show(block=False)
plt.close()
