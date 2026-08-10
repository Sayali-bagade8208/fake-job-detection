# Import the needed libraries.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the file paths with pathlib.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "top_industries_by_fake_jobs.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Treat missing/empty industry values safely
# ---------------------------------------------------
# Replace missing values and empty strings with "Unknown" for analysis only.
# We do not delete rows or edit the saved CSV file.
df["industry"] = df["industry"].fillna("Unknown")
df["industry"] = df["industry"].replace("", "Unknown")

# ---------------------------------------------------
# 4. Calculate industry-level counts and percentages
# ---------------------------------------------------
# Group by industry to summarize the target column.
industry_summary = df.groupby("industry").agg(
    total_postings=("fraudulent", "size"),
    real_jobs=("fraudulent", lambda x: (x == 0).sum()),
    fake_jobs=("fraudulent", lambda x: (x == 1).sum())
)

# Create a fake-job percentage column.
industry_summary["fake_job_percentage"] = (industry_summary["fake_jobs"] / industry_summary["total_postings"]) * 100

# Reset the index so we can work with a DataFrame table.
industry_summary = industry_summary.reset_index()

# Print the summary table.
print("Industry-level summary:")
print(industry_summary)
print()

# ---------------------------------------------------
# 5. Print top 15 industries by fake-job count
# ---------------------------------------------------
# Sort industries by fake job count from highest to lowest.
top_fake_count = industry_summary.sort_values("fake_jobs", ascending=False).head(15)

print("Top 15 industries by number of fake job postings:")
print(top_fake_count[["industry", "fake_jobs", "total_postings", "fake_job_percentage"]])
print()

# ---------------------------------------------------
# 6. Print top 15 industries by fake-job percentage
# ---------------------------------------------------
# Keep only industries with at least 30 total job postings.
filtered_for_percentage = industry_summary[industry_summary["total_postings"] >= 30]

# Sort by fake job percentage from highest to lowest.
top_fake_percentage = filtered_for_percentage.sort_values("fake_job_percentage", ascending=False).head(15)

print("Top 15 industries by fake-job percentage (minimum 30 postings):")
print(top_fake_percentage[["industry", "fake_jobs", "total_postings", "fake_job_percentage"]])
print()

# ---------------------------------------------------
# 7. Create a horizontal bar chart for top fake-count industries
# ---------------------------------------------------
# Select the top 15 industries by fake job count.
plot_data = top_fake_count.sort_values("fake_jobs", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(plot_data["industry"], plot_data["fake_jobs"], color="#d62728")

# Add labels and title.
plt.xlabel("Number of Fake Job Postings")
plt.ylabel("Industry")
plt.title("Top 15 Industries by Number of Fake Job Postings")
plt.tight_layout()

# Save the chart to a report folder.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without blocking the terminal.
plt.show(block=False)
plt.close()
