#include "NeuralNet.hpp"


Matrix nn::ReLULayer::forward (Matrix X) {
    this->result = (X > 0);
    return this->result * X;
}

Matrix nn::ReLULayer::backward (Matrix d_out) {
    Matrix d_result = d_out * this->result;
    return d_result;
}