#include "project2_a.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <string>

// Function to run a specific neural network architecture
void run_architecture(
    const std::vector<std::pair<unsigned, ActivationFunction*>>& layers_config,
    const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
    double learning_rate, double target_cost, unsigned max_iterations,
    double regularization_lambda,
    const std::string& cost_log_filename,
    const std::string& grid_output_filename,
    const std::string& arch_label) // New parameter for architecture label
{
    // Create a neural network with the given architecture
    unsigned input_size = 2;
    NeuralNetwork net(input_size, layers_config);

    // Train the network and log cost
    std::vector<double> cost_log;
    net.train(training_data, learning_rate, target_cost, max_iterations, cost_log, regularization_lambda);

    // Save cost log every 10,000 iterations
    std::ofstream cost_log_file(cost_log_filename);
    if (!cost_log_file.is_open()) {
        std::cerr << "Error: Could not open file " << cost_log_filename << " for writing." << std::endl;
        return;
    }

    // Determine the logging frequency
    // Original logging was every 50 iterations; now we log every 10,000 iterations
    // Therefore, we need to log every 200th entry (since 10,000 / 50 = 200)
    unsigned log_frequency = 200; // 200 * 50 = 10,000 iterations

    for (size_t i = 0; i < cost_log.size(); ++i) {
        // Check if the current index is at the desired logging frequency
        if ((i + 1) % log_frequency == 0) {
            unsigned current_iteration = (i + 1) * 50; // Each entry corresponds to 50 iterations
            cost_log_file << current_iteration << " " << cost_log[i] << "\n";
            // Include arch_label in the console output
            std::cout << "[" << arch_label << "] Iteration " << current_iteration << ": Cost = " << cost_log[i] << std::endl;
        }
    }
    cost_log_file.close();
    std::cout << "Cost log saved to " << cost_log_filename << "." << std::endl;

    // Evaluate network output on a grid, e.g., [0,1]x[0,1]
    std::ofstream grid_output_file(grid_output_filename);
    if (!grid_output_file.is_open()) {
        std::cerr << "Error: Could not open file " << grid_output_filename << " for writing." << std::endl;
        return;
    }

    double step = 0.01;
    for (double X1 = 0.0; X1 <= 1.0; X1 += step) {
        for (double X2 = 0.0; X2 <= 1.0; X2 += step) {
            DoubleVector input(2), output(1);
            input[0] = X1;
            input[1] = X2;
            net.feed_forward(input, output);
            grid_output_file << X1 << " " << X2 << " " << output[0] << "\n";
        }
    }
    grid_output_file.close();
    std::cout << "Grid output saved to " << grid_output_filename << "." << std::endl;
}

int main() {
    // Initialize activation function
    ActivationFunction* tanh_act = new TanhActivationFunction();

    // Load training data from 'spiral_training_data.dat'
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    std::ifstream training_file("spiral_training_data.dat");
    if (!training_file) {
        std::cerr << "Error: Could not open spiral_training_data.dat" << std::endl;
        delete tanh_act;
        return 1;
    }

    double x1, x2, label;
    while (training_file >> x1 >> x2 >> label) {
        DoubleVector input(2), output(1);
        input[0] = x1;
        input[1] = x2;
        output[0] = label;
        training_data.emplace_back(input, output);
    }
    training_file.close();

    std::cout << "Loaded " << training_data.size() << " training samples." << std::endl;

    // Training parameters
    double learning_rate = 0.01;
    double target_cost = 1e-3;
    unsigned max_iterations = 4000000;
    double regularization_lambda = 0.0;

    // Architecture 1: (2,4,4,1)
    {
        std::vector<std::pair<unsigned, ActivationFunction*>> layers_config = {
            {4, tanh_act}, {4, tanh_act}, {1, tanh_act}
        };
        run_architecture(layers_config, training_data, learning_rate, target_cost, max_iterations, regularization_lambda,
                         "cost_log_2_4_4_1.dat", "grid_output_2_4_4_1.dat", "Arch (2,4,4,1)");
    }

    // Architecture 2: (2,8,8,1)
    {
        std::vector<std::pair<unsigned, ActivationFunction*>> layers_config = {
            {8, tanh_act}, {8, tanh_act}, {1, tanh_act}
        };
        run_architecture(layers_config, training_data, learning_rate, target_cost, max_iterations, regularization_lambda,
                         "cost_log_2_8_8_1.dat", "grid_output_2_8_8_1.dat", "Arch (2,8,8,1)");
    }

    // Architecture 3: (2,16,16,1)
    {
        std::vector<std::pair<unsigned, ActivationFunction*>> layers_config = {
            {16, tanh_act}, {16, tanh_act}, {1, tanh_act}
        };
        run_architecture(layers_config, training_data, learning_rate, target_cost, max_iterations, regularization_lambda,
                         "cost_log_2_16_16_1.dat", "grid_output_2_16_16_1.dat", "Arch (2,16,16,1)");
    }

    // Clean up dynamically allocated memory
    delete tanh_act;
    return 0;
}
