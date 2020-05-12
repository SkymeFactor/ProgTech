#include "NeuralNet.hpp"
#include "Matrix.hpp"
#include <iostream>

using nn::Parameter;

nn::FCLayer::FCLayer (int n_input, int n_output) {
    this->W = Parameter(Matrix(n_input, n_output).fill_rand());
    this->B = Parameter(Matrix(1, n_output).fill_rand() * 0.001);
}

Matrix nn::FCLayer::forward (Matrix X) {
    this->X = X;
    return this->X.dot(this->W.value) + this->B.value;
}

Matrix nn::FCLayer::backward (Matrix d_out) {
    this->W.value.print(true);
    return d_out.dot(this->W.value.T());
}

std::pair<Parameter, Parameter> nn::FCLayer::get_params () {
    return std::pair<Parameter, Parameter> (this->W, this->B);
}