#include "NeuralNet.hpp"
#include "Matrix.hpp"
#include <math.h>

template <class T>
nn::AdamOptim<T>::AdamOptim (T beta_1, T beta_2, T epsilon) {
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->momentum = 0;
    this->velocity = 0;
    this->t = 1;
}

template <class T>
Matrix nn::AdamOptim<T>::update (Matrix w, Matrix d_w, T learning_rate){
    // Momentum update
    Matrix beta_momentum = this->momentum * this->beta_1;
    this->momentum = d_w * (1 - this->beta_1) + beta_momentum;
    // Velocity update
    Matrix beta_velocity = this->velocity * this->beta_2;
    this->velocity = ((d_w ^ 2.0) * (1 - this->beta_2)) + beta_velocity;

    Matrix m_hat = this->momentum / (1 - std::pow(this->beta_1, this->t));
    Matrix v_hat = this->velocity / (1 - std::pow(this->beta_2, this->t));

    this->t++;

    m_hat = (m_hat * learning_rate) / (v_hat.sqrt() + this->epsilon);

    return w - m_hat;
}

template class nn::AdamOptim<double>;