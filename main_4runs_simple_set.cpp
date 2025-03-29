#include "project2_a.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <sstream>

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

    // We will run the training process four times
    for (int run = 1; run <= 4; ++run) {
        // Create a new network for each run to ensure fresh parameters
        NeuralNetwork net(input_size, layers_config);

        std::vector<double> cost_log;
        net.train(training_data, learning_rate, target_cost, max_iterations, cost_log, regularization_lambda);

        // Save cost log with a unique filename per run
        {
            std::ostringstream fname;
            fname << "cost_log_run" << run << ".dat";
            std::ofstream cost_log_file(fname.str());
            for (size_t i = 0; i < cost_log.size(); ++i) {
                // Remember we now log every 50 iterations, so iteration index = i*50
                cost_log_file << i*50 << " " << cost_log[i] << std::endl;  
            }
        }
        std::cout << "Cost log for run " << run << " saved." << std::endl;

        // Evaluate network output on a grid for each run
        {
            std::ostringstream fname;
            fname << "grid_output_run" << run << ".dat";
            std::ofstream grid_output_file(fname.str());
            double step = 0.02; // or whatever step you use
            for (double X1 = -1.0; X1 <= 1.0; X1 += step) {
                for (double X2 = -1.0; X2 <= 1.0; X2 += step) {
                    DoubleVector input(2), output(1);
                    input[0] = X1;
                    input[1] = X2;
                    net.feed_forward(input, output);
                    grid_output_file << X1 << " " << X2 << " " << output[0] << std::endl;
                }
            }
            std::cout << "Grid output for run " << run << " saved." << std::endl;
        }

        std::cout << "Run " << run << " complete." << std::endl;
    }

    delete tanh_act;
    return 0;
}
