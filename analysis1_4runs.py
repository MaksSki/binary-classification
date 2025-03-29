#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script:
1. Safely loads and filters data from project_training_data.dat, cost_log_runN.dat, and grid_output_runN.dat.
2. Plots four runs' cost curves in one subplot.
3. Plots four runs' decision boundaries along with training data in another subplot.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import math

def load_data_file(filename, expected_cols):
    """
    Load a data file and filter out malformed lines.
    Returns a numpy array of shape (N, expected_cols).
    """
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            parts = line.split()
            if len(parts) != expected_cols:
                # Wrong number of columns, skip
                continue
            # Try to convert to float
            try:
                float_parts = [float(p) for p in parts]
                # If successful, add the line to our filtered list
                lines.append(" ".join(parts))
            except ValueError:
                # Non-numeric data found, skip this line
                continue

    if not lines:
        raise ValueError(f"No valid lines found in {filename}.")

    data_str = "\n".join(lines)
    data = np.loadtxt(StringIO(data_str))
    return data

def load_grid_data(filename):
    """
    Load grid data (should have 3 columns: x1, x2, z).
    Automatically determines the grid size and reshapes.
    """
    data = load_data_file(filename, 3)
    x1 = data[:,0]
    x2 = data[:,1]
    z = data[:,2]

    total_points = z.size
    num_points = int(math.isqrt(total_points))  # integer sqrt if Python 3.8+, else use int(np.sqrt(...))

    if num_points * num_points != total_points:
        raise ValueError(f"Grid is not square in {filename}. Total points: {total_points}, sqrt: {num_points}")

    X1 = x1.reshape(num_points, num_points)
    X2 = x2.reshape(num_points, num_points)
    Z = z.reshape(num_points, num_points)

    return X1, X2, Z

# ---------------------------------------------
# Load training data (3 columns: x1, x2, label)
# ---------------------------------------------
training_data = load_data_file('project_training_data.dat', 3)
X_train = training_data[:, 0]
Y_train = training_data[:, 1]
labels = training_data[:, 2]
mask_pos = (labels == 1)
mask_neg = (labels == -1)

# ---------------------------------------------
# Load and plot the cost data from four runs (2 cols: iteration, cost)
# ---------------------------------------------
runs = [1, 2, 3, 4]
cost_data_all = []
for run in runs:
    fname = f"cost_log_run{run}.dat"
    cost_data = load_data_file(fname, 2)
    cost_data_all.append(cost_data)

# ---------------------------------------------
# Load grid data for each run (3 cols: x1, x2, z)
# ---------------------------------------------
grid_data_runs = []
for run in runs:
    gname = f"grid_output_run{run}.dat"
    X1, X2, Z = load_grid_data(gname)
    grid_data_runs.append((X1, X2, Z))

# ---------------------------------------------
# Create a figure with two subplots
# ---------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the cost curves for each run on ax1
colors = ['b', 'g', 'r', 'm']
markers = ['o', 's', '^', 'd']
for i, run in enumerate(runs):
    data = cost_data_all[i]
    iterations = data[:, 0]
    cost = data[:, 1]
    ax1.plot(iterations, cost, marker=markers[i], linestyle='-', color=colors[i], label=f'Run {run}')
ax1.set_title('Cost over Iterations for Four Runs')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')
ax1.grid(True)
ax1.legend()

# Plot the decision boundaries from each run
line_styles = ['-', '--', '-.', ':']
for i, run in enumerate(runs):
    X1, X2, Z = grid_data_runs[i]
    # Draw contour at output=0
    cs = ax2.contour(X1, X2, Z, levels=[0.0], colors=[colors[i]], linestyles=[line_styles[i]], linewidths=2)

# Plot training points
ax2.scatter(X_train[mask_pos], Y_train[mask_pos], c='r', marker='o', label='+1')
ax2.scatter(X_train[mask_neg], Y_train[mask_neg], c='b', marker='o', label='-1')

ax2.set_title('Decision Boundaries & Training Data')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('four_runs_combined_plot.png', dpi=300)
plt.show()
