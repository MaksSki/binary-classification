import matplotlib.pyplot as plt
import pandas as pd

# Load the cost logs
original_log_file = "cost_log_original_algorithm.dat"
fixed_log_file = "cost_log_fixed_algorithm.dat"

# Load data into Pandas DataFrames
original_data = pd.read_csv(original_log_file, sep=" ", header=None, names=["Iteration", "Cost"])
fixed_data = pd.read_csv(fixed_log_file, sep=" ", header=None, names=["Iteration", "Cost"])

# Plot the cost logs
plt.figure(figsize=(10, 6))
plt.plot(original_data["Iteration"], original_data["Cost"], label="Original Algorithm", linewidth=2)
plt.plot(fixed_data["Iteration"], fixed_data["Cost"], label="Fixed Algorithm", linewidth=2)

# Customize the plot
plt.title("Cost Comparison: Original vs Fixed Algorithm", fontsize=16)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Save the plot
plt.savefig("cost_comparison_plot.png", dpi=300)

# Display the plot
plt.show()
