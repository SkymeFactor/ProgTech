#include "NeuralNet.hpp"
#include <math.h>
#include <iostream>


template <class T>
Matrix nn::SGD<T>::update (Matrix w, Matrix d_w, T learning_rate){
    Matrix half = d_w * learning_rate;
    return w - half;
}

template <class T>
std::shared_ptr<nn::Optim> nn::SGD<T>::copy () {
    return std::shared_ptr<nn::Optim>( new SGD(*this) );
}

template class nn::SGD<double>;