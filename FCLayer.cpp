#include "NeuralNet.hpp"

using nn::Parameter;

nn::FCLayer::FCLayer (int n_input, int n_output) {
    this->W = Parameter(Matrix(n_input, n_output).fill_rand() * 0.001);
    this->B = Parameter(Matrix(1, n_output).fill_rand() * 0.001);
}

Matrix nn::FCLayer::forward (Matrix &X) {
    this->X = X;
    Matrix result = this->X.dot(this->W.value);
    
    return result + this->B.value;
}

Matrix nn::FCLayer::backward (Matrix &d_out) {
    // W gradient computing
    Matrix w_grad = this->X.T().dot(d_out);
    this->W.grad = this->W.grad + w_grad;
    // B gradient computing
    Matrix b_grad = d_out.sum(0);
    this->B.grad = this->B.grad + b_grad;
    // Layer gradient computing
    return d_out.dot(this->W.value.T());
}

std::pair<Parameter*, Parameter*> nn::FCLayer::get_params () {
    return std::pair<Parameter*, Parameter*> (&(this->W), &(this->B));
}