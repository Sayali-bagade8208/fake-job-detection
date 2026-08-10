# Import the libraries needed for this beginner-friendly comparison script.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Define file paths
# ---------------------------------------------------
# Build the path to the reports folder where the chart image will be saved.
# This script reads metric values that were already obtained and does not retrain.
project_root = Path(__file__).resolve().parents[2]
reports_dir = project_root / "reports"
output_chart_path = reports_dir / "model_comparison.png"

# ---------------------------------------------------
# 2. Create the metric comparison table
# ---------------------------------------------------
# Build a DataFrame with the performance numbers already measured for each model.
comparison_data = {
    "Model": ["Logistic Regression", "LinearSVC"],
    "Accuracy": [0.9779082774049217, 0.9888143176733781],
    "Fake Precision": [0.7155963302752294, 0.9030303030303031],
    "Fake Recall": [0.9017341040462428, 0.861271676300578],
    "Fake F1": [0.7979539641943734, 0.8816568047337278],
    "ROC-AUC": [0.9877564678564816, 0.9911400855076871],
}

# Convert the data dictionary to a DataFrame for friendly printing.
comparison_df = pd.DataFrame(comparison_data)

# ---------------------------------------------------
# 3. Print the comparison table
# ---------------------------------------------------
print("Model comparison table")
print("=" * 80)
print(comparison_df.to_string(index=False))
print()

# ---------------------------------------------------
# 4. Build a grouped bar chart
# ---------------------------------------------------
# Set up a chart with one grouped bar for each metric.
# The x-axis will show the metric names.
# The y-axis is the metric value.
metric_columns = ["Accuracy", "Fake Precision", "Fake Recall", "Fake F1", "ROC-AUC"]

# Convert the metric rows into a layout that Matplotlib can draw.
# The DataFrame is already in a shape where each row is a model,
# and each column is a metric value.
plot_values = comparison_df.set_index("Model")[metric_columns]

# Create the grouped bar chart.
plot_values.plot(kind="bar", figsize=(10, 6), width=0.8)

# Add titles and axis labels.
plt.title("Fake Job Detection Model Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

# Make sure the reports folder exists.
reports_dir.mkdir(parents=True, exist_ok=True)

# Save the chart image.
plt.savefig(output_chart_path)

# Display the chart without blocking the terminal.
plt.show(block=False)
plt.close()

print(f"Saved comparison chart to: {output_chart_path}")
print("Model comparison complete. The CSV files were not modified.")
