#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:42:48 2024

@author: maksymilianskiba
"""

import matplotlib.pyplot as plt
import pandas as pd

# Load the timing data
timing_file = "timing_comparison.dat"

# Read the timing data
with open(timing_file, "r") as file:
    lines = file.readlines()
    original_time = float(lines[0].split(":")[1].strip().split()[0])  # Extract the time in ms
    fixed_time = float(lines[1].split(":")[1].strip().split()[0])

# Data for plotting
algorithms = ["Original Algorithm", "Fixed Algorithm"]
times = [original_time, fixed_time]

# Create the bar plot
plt.figure(figsize=(8, 6))
plt.bar(algorithms, times, color=['blue', 'green'], width=0.5)

# Add labels and title
plt.title("Runtime Comparison: Original vs Fixed Algorithm", fontsize=16)
plt.ylabel("Time (ms)", fontsize=14)
plt.xlabel("Algorithm", fontsize=14)



# Save the plot
plt.savefig("runtime_comparison_plot.png", dpi=300)

# Show the plot
plt.show()
