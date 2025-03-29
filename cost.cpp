#include "cost.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace BasicDenseLinearAlgebra;

void measure_costs(NeuralNetwork& net, const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data, const std::string& output_file) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << output_file << " for writing." << std::endl;
        return;
    }

    out << "SampleIndex,FiniteDifferencingTime,BackpropagationTime\n";

    for (size_t sample_idx = 0; sample_idx < training_data.size(); ++sample_idx) {
        const auto& [input, target] = training_data[sample_idx];

        // Measure finite differencing cost
        auto start_fd = std::chrono::high_resolution_clock::now();
        std::vector<DoubleMatrix> grad_w_fd(net.get_layer_count());
        std::vector<DoubleVector> grad_b_fd(net.get_layer_count());
        for (unsigned l = 0; l < net.get_layer_count(); ++l) {
            grad_w_fd[l] = DoubleMatrix(net.get_weights(l).n(), net.get_weights(l).m());
            grad_b_fd[l] = DoubleVector(net.get_biases(l).n());
        }
        net.compute_finite_difference(input, target, grad_w_fd, grad_b_fd);
        auto end_fd = std::chrono::high_resolution_clock::now();
        auto duration_fd = std::chrono::duration_cast<std::chrono::microseconds>(end_fd - start_fd).count();

        // Measure backpropagation cost
        auto start_bp = std::chrono::high_resolution_clock::now();
        std::vector<DoubleMatrix> grad_w_bp(net.get_layer_count());
        std::vector<DoubleVector> grad_b_bp(net.get_layer_count());
        for (unsigned l = 0; l < net.get_layer_count(); ++l) {
            grad_w_bp[l] = DoubleMatrix(net.get_weights(l).n(), net.get_weights(l).m());
            grad_b_bp[l] = DoubleVector(net.get_biases(l).n());
        }
        net.compute_backpropagation(input, target, grad_w_bp, grad_b_bp);
        auto end_bp = std::chrono::high_resolution_clock::now();
        auto duration_bp = std::chrono::duration_cast<std::chrono::microseconds>(end_bp - start_bp).count();

        // Output results to the file
        out << sample_idx << "," << duration_fd << "," << duration_bp << "\n";
    }

    out.close();
    std::cout << "Computational costs saved to " << output_file << std::endl;
}

int main() {
    // Load the training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    std::ifstream training_file("project_training_data.dat");
    if (!training_file.is_open()) {
        std::cerr << "Error: Could not open project_training_data.dat" << std::endl;
        return 1;
    }

    double x1, x2, label;
    while (training_file >> x1 >> x2 >> label) {
        DoubleVector input(2), target(1);
        input[0] = x1;
        input[1] = x2;
        target[0] = label;
        training_data.emplace_back(input, target);
    }
    training_file.close();

    std::cout << "Loaded " << training_data.size() << " training samples." << std::endl;

    // Define the network architecture
    std::vector<unsigned> layers = {4, 4, 1};
    NeuralNetwork net(2, layers);

    // Measure costs for finite differencing and backpropagation
    measure_costs(net, training_data, "computational_cost_comparison.csv");

    return 0;
}
