#include "NeuralNet.hpp"
#include <iostream>

std::pair<double, Matrix> nn::SoftmaxLayer::l2_reg (Matrix &W, const double &reg_strength){
    double loss = ((W ^ 2).sum(0).sum(1))(0, 0) * reg_strength;
    Matrix grad = W * reg_strength * 2;

    return std::pair<double, Matrix>(loss, grad);
}

Matrix nn::SoftmaxLayer::softmax (Matrix &predictions) {

    Matrix predictions_max = predictions.max(1);
    Matrix predictions_normal = predictions - predictions_max;
    Matrix predictions_exp = predictions_normal.exp();
    Matrix probs = predictions_exp / predictions_exp.sum(1);

    return probs;
}

// TODO: fix this function (Doesn't affect the result!)
double nn::SoftmaxLayer::ce_loss (Matrix &probs, Matrix &gt_index) {
    double loss;

    int shape = std::get<0>(gt_index.shape());
    Matrix loss_array(shape, std::get<0>(probs.shape()));
    for (int j = 0; j < std::get<0>(probs.shape()); j++ )
        for (int i = 0; i < shape; i++) {
            loss_array(i, j) = probs(j, gt_index(i, 0));
        }
    
    loss_array = - (loss_array.log()).mean(0).mean(1);
    loss = loss_array(0, 0);

    return loss;
}

std::pair<double, Matrix> nn::SoftmaxLayer::softmax_with_ce_loss(Matrix &predictions, Matrix &gt_index){
    Matrix zeroes(predictions.shape());

    zeroes.fill_zeros();

    // Marking the ground truth
    if (predictions.ndim() > 1) {
        for (int i = 0; i < std::get<0>(gt_index.shape()); i++){
            zeroes(i, gt_index(i, 0)) = 1;
        }
    } else {
        zeroes(0, gt_index(0, 0)) = 1;
    }

    // Compute sm and ce
    Matrix grad = softmax(predictions);
    double loss = ce_loss(grad, gt_index);
    grad = grad - zeroes;

    // Normalize grad
    if (predictions.ndim() > 1)
        grad = grad / std::get<0>(predictions.shape());

    return std::pair<double, Matrix>(loss, grad);
    
}