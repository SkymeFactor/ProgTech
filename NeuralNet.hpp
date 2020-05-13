#pragma once
#include <vector>
#include "Matrix.hpp"

/**********************************************************
 * nn - Neural Network namespace, contains all classes to create a simple perceptrone model.
 * --------------------------------------------------------
 * classes:
 *   Model - NN model, consist of some layers
 *   Parameter - Matrix based parameter for layers
 *   FCLayer - fully connected layer, parameters: w, b
 *   ReLULayer - ReLU layer, parameters: no params
 *   SoftmaxLayer - softmax, cross-entropy and l2 static functions
 *   AdamOptim - adam optimizer, parameters: beta1, beta2, epsilon
 * --------------------------------------------------------
 * Last changes 13 may 2020 by Skyme Factor.
 **********************************************************/
namespace nn {

    class Model {
    private:
        
    public:
        Model (int, int, int, double);
        double feed_forward (Matrix, Matrix);
        Matrix predict (Matrix);
        std::vector<Matrix> get_params();
    };

    class Parameter {
    public:
        Matrix value;
        Matrix grad;
        Parameter () {};
        explicit Parameter (Matrix value) {
            this->value = value;
            this->grad = Matrix(value.shape());
        }
    };

    class FCLayer {
    private:
        Parameter W, B;
        Matrix X;
    public:
        explicit FCLayer (int n_input, int n_output);
        Matrix forward (Matrix X);
        Matrix backward (Matrix d_out);
        std::pair<Parameter, Parameter> get_params ();
    };

    class ReLULayer {
    private:
        Matrix result;
    public:
        ReLULayer () {};
        Matrix forward (Matrix X);
        Matrix backward (Matrix d_out);
    };

    class SoftmaxLayer {
    public:
        static std::pair<double, Matrix> l2_reg (Matrix &, const double &);
        static Matrix softmax (Matrix &);
        static double ce_loss (Matrix &, Matrix &);
        static std::pair<double, Matrix> softmax_with_ce_loss (Matrix &, Matrix &);
    };

    template <class T>
    class AdamOptim {
    private:
        T beta_1;
        T beta_2;
        T epsilon;
        Matrix momentum;
        Matrix velocity;
        T t;
    public:
        /*
        * Explicit constructor of class AdamOptim
        * Parameters:
        *   <T> beta1 - default 0.9
        *   <T> beta2 - default 0.999
        *   <T> epsilon - default 1e-8
        */
        explicit AdamOptim<T> (T beta_1 = 0.9, T beta_2 = 0.999, T epsilon = 1e-08);
        /*
        * Parameters:
        *   Matrix w - model parameter
        *   Matrix d_w - gradient for parameter
        *   <T> learning_rate - learning rate for the model
        */
        Matrix update (Matrix w, Matrix d_w, T learning_rate);
    };
}