#include "NeuralNet.hpp"


Matrix nn::ReLULayer::forward (Matrix &X) {
    this->X = (X > 0);
    return this->X * X;
}

Matrix nn::ReLULayer::backward (Matrix &d_out) {
    Matrix d_result = d_out * this->X;
    return d_result;
}