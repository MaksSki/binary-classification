#pragma once

#include "project2_a_basics.h"
#include "dense_linear_algebra.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <stdexcept>


using namespace BasicDenseLinearAlgebra;

// Forward declarations
inline DoubleVector multiply(const DoubleMatrix& mat, const DoubleVector& vec);
inline DoubleMatrix transpose(const DoubleMatrix& mat);
inline DoubleMatrix outer_product(const DoubleVector& a, const DoubleVector& b);

// Neural Network Layer
class NeuralNetworkLayer {
public:
    NeuralNetworkLayer(unsigned input_size, unsigned output_size, ActivationFunction* act_func)
        : weights(output_size, input_size), biases(output_size), activation_function(act_func) {}

    DoubleVector forward(const DoubleVector& input, DoubleVector& z) const {
        z = DoubleVector(weights.n());
        DoubleVector output(weights.n());
        for (unsigned i = 0; i < weights.n(); ++i) {
            double sum = biases[i];
            for (unsigned j = 0; j < weights.m(); ++j) {
                sum += weights(i, j) * input[j];
            }
            z[i] = sum;
            output[i] = activation_function->sigma(sum);
        }
        return output;
    }

    DoubleMatrix& get_weights() { return weights; }
    DoubleVector& get_biases() { return biases; }
    ActivationFunction* get_activation_function() const { return activation_function; }

private:
    DoubleMatrix weights;
    DoubleVector biases;
    ActivationFunction* activation_function;
};

// Neural Network
class NeuralNetwork : public NeuralNetworkBasis {
public:
    NeuralNetwork(unsigned input_size, const std::vector<std::pair<unsigned, ActivationFunction*>>& layers_config) {
        unsigned prev_size = input_size;
        for (auto& [size, act] : layers_config) {
            layers.emplace_back(prev_size, size, act);
            prev_size = size;
        }
    }

    void feed_forward(const DoubleVector& input, DoubleVector& output) const override {
        DoubleVector activation = input;
        for (auto& layer : layers) {
            DoubleVector z;
            activation = layer.forward(activation, z);
        }
        output = activation;
    }

    double cost(const DoubleVector& input, const DoubleVector& target_output) const override {
        DoubleVector output;
        feed_forward(input, output);
        double cost_val = 0.0;
        for (unsigned i = 0; i < output.n(); ++i) {
            double diff = output[i] - target_output[i];
            cost_val += 0.5 * diff * diff;
        }
        return cost_val;
    }

    double cost_for_training_data(const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) const override {
        double total_cost = 0.0;
        for (const auto& [input, target] : training_data) {
            total_cost += cost(input, target);
        }
        return total_cost / training_data.size();
    }

    void initialise_parameters() {
        std::mt19937& gen = RandomNumber::Random_number_generator;
        std::normal_distribution<double> dist(0.0, 0.1);

        for (auto& layer : layers) {
            for (unsigned i = 0; i < layer.get_weights().n(); ++i) {
                for (unsigned j = 0; j < layer.get_weights().m(); ++j) {
                    layer.get_weights()(i, j) = dist(gen);
                }
            }

            for (unsigned i = 0; i < layer.get_biases().n(); ++i) {
                layer.get_biases()[i] = dist(gen);
            }
        }
    }

    void train(const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
               double learning_rate, double target_cost, unsigned max_iterations,
               std::vector<double>& cost_log, double regularization_lambda) {
        initialise_parameters();
        unsigned iteration = 0;
        double current_cost = cost_for_training_data(training_data);

        while (current_cost > target_cost && iteration < max_iterations) {
            for (const auto& [input, target] : training_data) {
                std::vector<DoubleMatrix> grad_w;
                std::vector<DoubleVector> grad_b;
                grad_w.reserve(layers.size());
                grad_b.reserve(layers.size());
                for (unsigned l = 0; l < layers.size(); ++l) {
                    grad_w.emplace_back(layers[l].get_weights().n(), layers[l].get_weights().m());
                    grad_b.emplace_back(layers[l].get_biases().n());
                }

                backpropagation(input, target, grad_w, grad_b);

                // Update parameters
                for (unsigned l = 0; l < layers.size(); ++l) {
                    auto& weights = layers[l].get_weights();
                    auto& biases = layers[l].get_biases();

                    for (unsigned i = 0; i < weights.n(); ++i) {
                        for (unsigned j = 0; j < weights.m(); ++j) {
                            weights(i, j) -= learning_rate * (grad_w[l](i, j) + regularization_lambda * weights(i, j));
                        }
                    }

                    for (unsigned i = 0; i < biases.n(); ++i) {
                        biases[i] -= learning_rate * grad_b[l][i];
                    }
                }
            }

            // Log cost every 50 iterations
            if (iteration % 50 == 0) {
                current_cost = cost_for_training_data(training_data);
                cost_log.push_back(current_cost);
                std::cout << "Iteration " << iteration << ": Cost = " << current_cost << std::endl;
            }

            ++iteration;
        }

        if (current_cost <= target_cost) {
            std::cout << "Training converged successfully after " << iteration << " iterations." << std::endl;
        } else {
            std::cout << "Training stopped after reaching the maximum number of iterations." << std::endl;
        }
    }

    void backpropagation(const DoubleVector& input, const DoubleVector& target,
                         std::vector<DoubleMatrix>& grad_w, std::vector<DoubleVector>& grad_b) {
        std::vector<DoubleVector> activations, zs;
        DoubleVector activation = input;
        activations.push_back(activation);

        // Forward pass
        for (auto& layer : layers) {
            DoubleVector z;
            activation = layer.forward(activation, z);
            zs.push_back(z);
            activations.push_back(activation);
        }

        // Backward pass: output layer
        DoubleVector delta = activations.back();
        for (unsigned i = 0; i < delta.n(); ++i) {
            delta[i] -= target[i]; // delta = a^(L) - y
        }

        // Apply derivative of activation at output layer
        for (unsigned i = 0; i < delta.n(); ++i) {
            double dz = layers.back().get_activation_function()->dsigma(zs.back()[i]);
            delta[i] *= dz;
        }

        grad_w.back() = outer_product(delta, activations[activations.size() - 2]);
        grad_b.back() = delta;

        // Hidden layers
        for (int l = (int)layers.size() - 2; l >= 0; --l) {
            delta = multiply(transpose(layers[l + 1].get_weights()), delta);

            for (unsigned i = 0; i < delta.n(); ++i) {
                double dz = layers[l].get_activation_function()->dsigma(zs[l][i]);
                delta[i] *= dz;
            }

            grad_w[l] = outer_product(delta, activations[l]);
            grad_b[l] = delta;
        }
    }

    // Measure finite differencing cost
    void compute_finite_difference() {
        // Placeholder for finite-differencing functionality
    }

    // Measure backpropagation cost
    void compute_backpropagation_cost() {
        // Placeholder for backpropagation functionality
    }

private:
    std::vector<NeuralNetworkLayer> layers;
};

// Helper functions
inline DoubleVector multiply(const DoubleMatrix& mat, const DoubleVector& vec) {
    if (mat.m() != vec.n()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match.");
    }

    DoubleVector result(mat.n());
    for (unsigned i = 0; i < mat.n(); ++i) {
        double sum = 0.0;
        for (unsigned j = 0; j < mat.m(); ++j) {
            sum += mat(i, j) * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

inline DoubleMatrix transpose(const DoubleMatrix& mat) {
    DoubleMatrix result(mat.m(), mat.n());
    for (unsigned i = 0; i < mat.n(); ++i) {
        for (unsigned j = 0; j < mat.m(); ++j) {
            result(j, i) = mat(i, j);
        }
    }
    return result;
}

inline DoubleMatrix outer_product(const DoubleVector& a, const DoubleVector& b) {
    DoubleMatrix result(a.n(), b.n());
    for (unsigned i = 0; i < a.n(); ++i) {
        for (unsigned j = 0; j < b.n(); ++j) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}
