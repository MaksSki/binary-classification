#include "project2_a.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <sstream>
#include <chrono>

int main() {
    ActivationFunction* tanh_act = new TanhActivationFunction();

    // Define the network structure: (2,3,3,1)
    unsigned input_size = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> layers_config = {
        {3, tanh_act}, {3, tanh_act}, {1, tanh_act}
    };

    // Load training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    std::ifstream training_file("project_training_data.dat");
    if (!training_file) {
        std::cerr << "Error: Could not open training data file." << std::endl;
        return 1;
    }

    double x1, x2, y;
    while (training_file >> x1 >> x2 >> y) {
        DoubleVector input(2), output(1);
        input[0] = x1;
        input[1] = x2;
        output[0] = y;
        training_data.emplace_back(input, output);
    }
    training_file.close();

    std::cout << "Loaded " << training_data.size() << " training samples." << std::endl;

    // Training parameters
    double learning_rate = 0.1;
    double target_cost = 1e-4;
    unsigned max_iterations = 100000;
    double regularization_lambda = 0.0; 

    // Run original algorithm
    std::cout << "Running original algorithm..." << std::endl;
    NeuralNetwork original_net(input_size, layers_config);

    std::vector<double> original_cost_log;
    auto start_original = std::chrono::high_resolution_clock::now();
    original_net.train(training_data, learning_rate, target_cost, max_iterations, original_cost_log, regularization_lambda);
    auto end_original = std::chrono::high_resolution_clock::now();
    auto duration_original = std::chrono::duration_cast<std::chrono::milliseconds>(end_original - start_original).count();

    // Save original cost log
    {
        std::ofstream cost_log_file("cost_log_original_algorithm.dat");
        for (size_t i = 0; i < original_cost_log.size(); ++i) {
            cost_log_file << i * 50 << " " << original_cost_log[i] << std::endl;
        }
    }
    std::cout << "Original algorithm completed in " << duration_original << " ms. Cost log saved." << std::endl;

    // Run fixed algorithm
    std::cout << "Running fixed (interlaced) algorithm..." << std::endl;
    NeuralNetwork fixed_net(input_size, layers_config);

    std::vector<double> fixed_cost_log;
    auto start_fixed = std::chrono::high_resolution_clock::now();
    fixed_net.train(training_data, learning_rate, target_cost, max_iterations, fixed_cost_log, regularization_lambda);
    auto end_fixed = std::chrono::high_resolution_clock::now();
    auto duration_fixed = std::chrono::duration_cast<std::chrono::milliseconds>(end_fixed - start_fixed).count();

    // Save fixed cost log
    {
        std::ofstream cost_log_file("cost_log_fixed_algorithm.dat");
        for (size_t i = 0; i < fixed_cost_log.size(); ++i) {
            cost_log_file << i * 50 << " " << fixed_cost_log[i] << std::endl;
        }
    }
    std::cout << "Fixed algorithm completed in " << duration_fixed << " ms. Cost log saved." << std::endl;

    // Save timing comparison
    {
        std::ofstream timing_file("timing_comparison.dat");
        timing_file << "Original Algorithm: " << duration_original << " ms\n";
        timing_file << "Fixed Algorithm: " << duration_fixed << " ms\n";
    }
    std::cout << "Timing comparison saved to timing_comparison.dat." << std::endl;

    delete tanh_act;
    return 0;
}
