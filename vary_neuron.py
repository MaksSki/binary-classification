import numpy as np
import matplotlib.pyplot as plt

# Plot the cost function
def plot_costs(filenames, labels, output_filename):
    plt.figure(figsize=(10, 6))
    for filename, label in zip(filenames, labels):
        data = np.loadtxt(filename)
        iterations = data[:, 0]
        cost = data[:, 1]
        plt.plot(iterations, cost, label=label)
    plt.title('Cost over Iterations for Architectures (4, 8, 16)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

# Plot decision boundaries
def plot_decision_boundary(grid_filenames, training_data_file, labels, output_filename):
    plt.figure(figsize=(18, 5))

    # Load training data (handling inconsistent columns)
    with open(training_data_file, 'r') as f:
        training_data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Only process rows with exactly 3 columns
                training_data.append([float(x) for x in parts])
    training_data = np.array(training_data)

    X_train = training_data[:, 0]
    Y_train = training_data[:, 1]
    labels_train = training_data[:, 2]
    mask_pos = (labels_train == 1)
    mask_neg = (labels_train == -1)

    for idx, (grid_filename, label) in enumerate(zip(grid_filenames, labels)):
        grid_data = np.loadtxt(grid_filename)
        x1 = grid_data[:, 0]
        x2 = grid_data[:, 1]
        z = grid_data[:, 2]

        # Determine grid size
        num_points = int(np.sqrt(z.size))
        X1 = x1.reshape(num_points, num_points)
        X2 = x2.reshape(num_points, num_points)
        Z = z.reshape(num_points, num_points)

        # Plot decision boundary
        plt.subplot(1, 3, idx + 1)
        plt.contour(X1, X2, Z, levels=[0.0], colors='k')
        plt.scatter(X_train[mask_pos], Y_train[mask_pos], c='r', label='+1')
        plt.scatter(X_train[mask_neg], Y_train[mask_neg], c='b', label='-1')
        plt.title(f'Decision Boundary for Architecture {label}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    # Filenames for cost logs and grid outputs
    cost_files = [
        "cost_log_4.dat", 
        "cost_log_8.dat", 
        "cost_log_16.dat"
    ]
    grid_files = [
        "grid_output_4.dat", 
        "grid_output_8.dat", 
        "grid_output_16.dat"
    ]
    training_data_file = "spiral_training_data.dat"

    # Labels for the architectures
    cost_labels = ["2-4-4-1", "2-8-8-1", "2-16-16-1"]

    # Plot cost functions
    plot_costs(cost_files, cost_labels, "cost_comparison_4_8_16.png")

    # Plot decision boundaries
    plot_decision_boundary(grid_files, training_data_file, cost_labels, "decision_boundaries_4_8_16.png")

    # Combine the two plots into a single visualization (if needed)
    plt.figure(figsize=(12, 6))
    for filename, label in zip(cost_files, cost_labels):
        data = np.loadtxt(filename)
        iterations = data[:, 0]
        cost = data[:, 1]
        plt.plot(iterations, cost, label=f'Cost - {label}')
    plt.title('Combined Cost and Decision Boundaries')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_plot_4_8_16.png")
    plt.show()
