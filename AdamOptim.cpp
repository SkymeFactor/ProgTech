#include "NeuralNet.hpp"
#include <Matrix.hpp>

template <class T>
nn::AdamOptim<T>::AdamOptim (T beta_1 = 0.9, T beta_2 = 0.999, T epsilon = 1e-08) {
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->velocity = 0;
    this->t = 1;
}

template <class T>
Matrix nn::AdamOptim<T>::update (Matrix w, Matrix d_w, T learning_rate){

    this->momentum = this->beta_1 * this->momentum + (1 - this->beta_2) * d_w;
    this->velocity = this->beta_2 * this->velocity + (1 - this->beta_2) * d_w ^ 2;

    Matrix m_hat;
    Matrix v_hat;

    this->t++;

    return w - learning_rate * m_hat / (v_hat.sqrt() + this->epsilon);
}