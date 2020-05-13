#include "NeuralNet.hpp"

std::pair<double, Matrix> nn::SoftmaxLayer::l2_reg (Matrix &W, const double &reg_strength){
    double loss = ((W ^ 2).sum(0).sum(1))(0, 0) * reg_strength;
    Matrix grad = W * reg_strength * 2;

    return std::pair(loss, grad);
}

Matrix nn::SoftmaxLayer::softmax (Matrix &predictions) {
    int p_dim = predictions.ndim();

    Matrix predictions_max = predictions.max(1);
    predictions = predictions - predictions_max;

    Matrix probs = predictions.exp() / (predictions.exp()).sum(1);

    return probs;
}

double nn::SoftmaxLayer::ce_loss (Matrix &probs, Matrix &gt_index) {
    double loss;
    int shape = std::get<1>(gt_index.shape());
    Matrix loss_array(1, shape);
    for (int i = 0; i < shape; i++)
        loss_array(0, i) = probs(i, gt_index(0, i));
    loss_array = - (loss_array.log()).mean(1);
    loss = loss_array(0, 0);

    return loss;
}

std::pair<double, Matrix> nn::SoftmaxLayer::softmax_with_ce_loss(Matrix &predictions, Matrix &gt_index){
    Matrix zeroes(predictions.shape());

    // Marking the ground truth
    if (predictions.ndim() > 1) {
        for (int i = 0; i < std::get<1>(gt_index.shape()); i++)
            zeroes(i, gt_index(0, i)) = 1;
    } else {
        zeroes(0, gt_index(0, 0)) = 1;
    }

    Matrix grad = softmax(predictions);
    double loss = ce_loss(grad, gt_index);
    grad = grad - zeroes;

    if (predictions.ndim() > 1)
        grad = grad / std::get<0>(predictions.shape());

    return std::pair(loss, grad);
    
}