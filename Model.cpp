#include "NeuralNet.hpp"
#include <iostream>

// This constructor was supposed to accept layers as parameters, but it doesn't matter anyway
nn::Model::Model(const int &n_input, const int &n_output, const int &n_hidden, const double &reg) {
    this->reg = reg;
    // Push the layers to stack
    layers.push_back(new FCLayer(n_input, n_hidden));
    layers.push_back(new ReLULayer());
    layers.push_back(new FCLayer(n_hidden, n_output));
}

double nn::Model::feed_forward (Matrix &X, Matrix &y) {

    // Get and nullify parameters
    auto params = this->get_params();
    for (int i = 0; i < (int)params.size(); i++)
        params[i]->grad = params[i]->grad.fill_zeros();
    
    // Feed forward
    Matrix temp = X;
    for (auto it : layers){
        temp = (*it).forward(temp);
    }

    // Compute sm and ce loss
    auto result = sml.softmax_with_ce_loss(temp, y);

    // Back propagation
    temp = result.second;    
    for (auto it = layers.rbegin(); it != layers.rend(); it++) {
        temp = (*it)->backward(temp);
    }

    // Regularization
    for (int i = 0; i < (int)params.size(); i++) {
        auto temp = sml.l2_reg(params[i]->value, this->reg);
        result.first += temp.first;
        params[i]->grad = params[i]->grad + temp.second;
    }

    // Return loss
    return result.first;
}

Matrix nn::Model::predict (Matrix &X) {
    // Forward prop
    Matrix temp = X;
    for (auto it : layers){
        temp = (*it).forward(temp);
    }

    // Extract predictions
    Matrix pred = temp.argmax(1);

    return pred;
}

vector<nn::Parameter*> nn::Model::get_params () {
    vector<nn::Parameter*> result;

    for (auto it : layers)
        if (typeid(*it) == typeid(FCLayer)){
            FCLayer *pt = (FCLayer *)it;
            auto temp = (*pt).get_params();
            result.push_back(temp.first);
            result.push_back(temp.second);
        }
    
    return result;
}