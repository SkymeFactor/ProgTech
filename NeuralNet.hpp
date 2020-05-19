#pragma once
#include <vector>
#include <memory>
#include "MatrixLib/Matrix.hpp"

/**********************************************************
 * nn - Neural Network namespace, contains all classes to create a simple perceptrone model.
 * --------------------------------------------------------
 * classes:
 *   Parameter - Matrix based parameter for layers
 *   Layer - base class for all layers except SoftmaxLayer
 *   FCLayer - fully connected layer, parameters: w, b
 *   ReLULayer - ReLU layer, parameters: no params
 *   SoftmaxLayer - softmax, cross-entropy and l2 static functions
 *   SGD - adam optimizer, parameters: beta1, beta2, epsilon
 *   Model - NN model, consist of some layers
 * --------------------------------------------------------
 * Last changes 13 may 2020 by Skyme Factor.
 **********************************************************/
namespace nn {

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

    class Layer {
    protected:
        Matrix X;
        virtual ~Layer() {};
    public:
        virtual Matrix forward (Matrix &) = 0;
        virtual Matrix backward (Matrix &) = 0;
        //virtual std::pair<Parameter, Parameter> get_params () = 0;
    };

    class FCLayer : public Layer {
    private:
        Parameter W, B;
        Matrix X;
    public:
        explicit FCLayer (int n_input, int n_output);
        virtual Matrix forward (Matrix &X) override;
        virtual Matrix backward (Matrix &d_out) override;
        std::pair<Parameter*, Parameter*> get_params ();
    };

    class ReLULayer : public Layer {
    private:
        Matrix X;
    public:
        ReLULayer () {};
        virtual Matrix forward (Matrix &X) override;
        virtual Matrix backward (Matrix &d_out) override;
        //virtual std::pair<Parameter, Parameter> get_params () override;
    };

    class SoftmaxLayer {
    public:
        static std::pair<double, Matrix> l2_reg (Matrix &, const double &);
        static Matrix softmax (Matrix &);
        static double ce_loss (Matrix &, Matrix &);
        static std::pair<double, Matrix> softmax_with_ce_loss (Matrix &, Matrix &);
    };

    class Optim {
    protected:
        virtual ~Optim () {};
    public:
        virtual Matrix update (Matrix , Matrix , double) = 0;
        virtual std::shared_ptr<Optim> copy () = 0;
    };

    template <class T>
    class SGD : public Optim {
    private:
        T beta_1;
        T beta_2;
        T epsilon;
        Matrix momentum;
        Matrix velocity;
        T t;
    public:
        /*
        * Explicit constructor of class SGD
        * Parameters:
        *   No parameters.
        */
        explicit SGD<T> () {};
        /*
        * Parameters:
        *   Matrix w - model parameter
        *   Matrix d_w - gradient for parameter
        *   <T> learning_rate - learning rate for the model
        */
        Matrix update (Matrix w, Matrix d_w, T learning_rate);
        // This is used to make possible copying by pointer
        std::shared_ptr<Optim> copy ();
    };

    class Model {
    private:
        double reg;
        vector<Layer *> layers;
        SoftmaxLayer sml;
    public:
        /*
        * Explicit constructor of class Model
        * Parameters:
        *  const int &n_input - input layer size
        *  const int &n_output - number of classes to predict
        *  const int &n_hidden - hidden layer(s) size
        *  const double &reg - regularization strength
        */
        Model () {};
        explicit Model (const int &, const int &, const int &, const double &);
        double feed_forward (Matrix &, Matrix &);
        Matrix predict (Matrix &);
        std::vector<Parameter*> get_params();
    };

    #define DATASET_TYPE std::pair<std::vector<std::vector<double>>, std::vector<double>>

    class Trainer {
    private:
        nn::Model model;
        DATASET_TYPE dataset;
        nn::Optim* optim;
        int num_epochs;
        int batch_size;
        double learning_rate;
        double learning_rate_decay;
    public:
        explicit Trainer (nn::Model &, DATASET_TYPE &, nn::Optim*, int = 20, int = 20, double = 1e-2, double = 1.0);
        vector<vector<int>> split_indices (vector<int> , int, bool = true);
        std::vector<std::vector<double>> fit ();
        static double compute_accuracy (Matrix &, Matrix &);
    };
}