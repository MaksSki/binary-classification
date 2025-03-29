import numpy as np
import matplotlib.pyplot as plt

# Load cost log files
cost_arch1 = np.loadtxt('cost_log_arch1.dat')
cost_arch2 = np.loadtxt('cost_log_arch2.dat')
cost_arch3 = np.loadtxt('cost_log_arch3.dat')

# Load grid output data for decision boundaries
grid_arch1 = np.loadtxt('grid_output_arch1.dat')
grid_arch2 = np.loadtxt('grid_output_arch2.dat')
grid_arch3 = np.loadtxt('grid_output_arch3.dat')

# Extract cost data
iterations_arch1 = cost_arch1[:, 0]
cost_values_arch1 = cost_arch1[:, 1]

iterations_arch2 = cost_arch2[:, 0]
cost_values_arch2 = cost_arch2[:, 1]

iterations_arch3 = cost_arch3[:, 0]
cost_values_arch3 = cost_arch3[:, 1]

# Extract grid data
x1_arch1 = grid_arch1[:, 0]
x2_arch1 = grid_arch1[:, 1]
z_arch1 = grid_arch1[:, 2]

x1_arch2 = grid_arch2[:, 0]
x2_arch2 = grid_arch2[:, 1]
z_arch2 = grid_arch2[:, 2]

x1_arch3 = grid_arch3[:, 0]
x2_arch3 = grid_arch3[:, 1]
z_arch3 = grid_arch3[:, 2]

# Determine grid size
num_points = int((1 - 0) / 0.01)  # Grid step is 0.01

X1_arch1 = x1_arch1.reshape(num_points, num_points)
X2_arch1 = x2_arch1.reshape(num_points, num_points)
Z_arch1 = z_arch1.reshape(num_points, num_points)

X1_arch2 = x1_arch2.reshape(num_points, num_points)
X2_arch2 = x2_arch2.reshape(num_points, num_points)
Z_arch2 = z_arch2.reshape(num_points, num_points)

X1_arch3 = x1_arch3.reshape(num_points, num_points)
X2_arch3 = x2_arch3.reshape(num_points, num_points)
Z_arch3 = z_arch3.reshape(num_points, num_points)

# Load training data and handle inconsistent rows
training_data = []
with open('spiral_training_data.dat', 'r') as file:
    for line in file:
        parts = line.split()
        if len(parts) == 3:  # Only process rows with exactly 3 columns
            training_data.append([float(value) for value in parts])

training_data = np.array(training_data)
X_train = training_data[:, 0]
Y_train = training_data[:, 1]
labels = training_data[:, 2]

mask_pos = (labels == 1)
mask_neg = (labels == -1)

# Plot cost over iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations_arch1, cost_values_arch1, label='Architecture 1 (2, 4, 4, 1)', linestyle='-', marker='o')
plt.plot(iterations_arch2, cost_values_arch2, label='Architecture 2 (2, 4, 4, 1)', linestyle='--', marker='s')
plt.plot(iterations_arch3, cost_values_arch3, label='Architecture 3 (2, 4, 4, 4, 1)', linestyle='-.', marker='^')
plt.title('Cost over Iterations for Different Architectures')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cost_plot.png', dpi=300)
plt.show()

# Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot decision boundaries for Architecture 1
cs1 = axes[0].contour(X1_arch1, X2_arch1, Z_arch1, levels=[0.0], colors='k', linewidths=2)
axes[0].scatter(X_train[mask_pos], Y_train[mask_pos], c='r', marker='o', label='+1')
axes[0].scatter(X_train[mask_neg], Y_train[mask_neg], c='b', marker='o', label='-1')
axes[0].set_title('Decision Boundary: Architecture 1 (2, 4, 4, 1)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].legend()
axes[0].grid(True)

# Plot decision boundaries for Architecture 2
cs2 = axes[1].contour(X1_arch2, X2_arch2, Z_arch2, levels=[0.0], colors='k', linewidths=2)
axes[1].scatter(X_train[mask_pos], Y_train[mask_pos], c='r', marker='o', label='+1')
axes[1].scatter(X_train[mask_neg], Y_train[mask_neg], c='b', marker='o', label='-1')
axes[1].set_title('Decision Boundary: Architecture 2 (2, 4, 4, 4, 1)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].legend()
axes[1].grid(True)

# Plot decision boundaries for Architecture 3
cs3 = axes[2].contour(X1_arch3, X2_arch3, Z_arch3, levels=[0.0], colors='k', linewidths=2)
axes[2].scatter(X_train[mask_pos], Y_train[mask_pos], c='r', marker='o', label='+1')
axes[2].scatter(X_train[mask_neg], Y_train[mask_neg], c='b', marker='o', label='-1')
axes[2].set_title('Decision Boundary: Architecture 3 (2, 4, 4, 4, 1)')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=300)
plt.show()
