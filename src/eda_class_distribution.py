# Import the needed libraries.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define the dataset and report paths
# ---------------------------------------------------
# Use pathlib so the script can locate files safely.
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "data" / "cleaned_job_postings.csv"
output_chart_path = project_root / "reports" / "class_distribution.png"

# ---------------------------------------------------
# 2. Load the cleaned dataset
# ---------------------------------------------------
# Read the cleaned CSV using Pandas.
df = pd.read_csv(input_path)

# ---------------------------------------------------
# 3. Calculate class counts and percentages
# ---------------------------------------------------
# Count how many rows have each fraudulent label.
class_counts = df["fraudulent"].value_counts()

# Sort by label so the display order is 0 then 1.
class_counts = class_counts.sort_index()

# Calculate the percentage for each class.
class_percentages = (class_counts / len(df)) * 100

# Print the class distribution.
print("Class counts:")
print(class_counts)
print()

print("Class percentages:")
print(class_percentages)
print()

# ---------------------------------------------------
# 4. Create the bar chart
# ---------------------------------------------------
# Define labels for the two classes.
labels = ["Real Jobs (0)", "Fake Jobs (1)"]

# The class_counts may be indexed by 0 and 1.
# Create a list of values in the same order as the labels.
values = [class_counts.get(0, 0), class_counts.get(1, 0)]

# Create a simple bar chart.
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=["#2ca02c", "#d62728"])

# Add labels and title.
plt.xlabel("Job Class")
plt.ylabel("Number of Jobs")
plt.title("Fake Job Detection - Class Distribution")

# Add count labels above each bar.
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        str(int(height)),
        ha="center",
        va="bottom"
    )

# Make the chart layout tidy.
plt.tight_layout()

# Save the chart to a report folder.
output_chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_chart_path)

# Display the chart without waiting for a GUI event loop.
plt.show(block=False)
plt.close()
