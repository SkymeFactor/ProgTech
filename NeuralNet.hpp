#pragma once
#include "Matrix.hpp"

namespace nn {

    class Model {
    private:
        
    public:

    };

    class Parameter {
    private:
        Matrix value;
        Matrix grad;
    public:
        Parameter (Matrix value) {
            this->value = value;
            this->grad = Matrix(value.shape());
        }
    };

    class FCLayer {
    private:
        Matrix W, B, X;
    public:
        FCLayer (int n_input, int n_output);
        Matrix forward (Matrix X);
        Matrix backward (Matrix d_out);
        std::pair<Matrix, Matrix> get_params ();
    };

    class ReLULayer {
    private:
        Matrix result;
    public:
        ReLULayer ();
        Matrix forward (Matrix X);
        Matrix backward (Matrix d_out);
    };

    template <class T>
    class AdamOptim {
    private:
        T beta_1;
        T beta_2;
        T epsilon;
        T momentum;
        T velocity;
        int t;
    public:
        explicit AdamOptim<T> (T beta_1, T beta_2, T epsilon);
        Matrix update (Matrix w, Matrix d_w, T learning_rate);
    };
}