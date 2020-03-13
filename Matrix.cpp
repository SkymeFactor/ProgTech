#include "Matrix.hpp"

#include <iostream>
#include <vector>
#include <tuple>
#include <math.h>

using namespace std;

Matrix::Matrix(){
    this->row_size = 1;
    this->col_size = 1;
    this->matrix.resize(this->row_size);
    this->matrix[0].resize(this->col_size, 0.0);
};

Matrix::Matrix (int rows, int cols) {
    this->row_size = rows;
    this->col_size = cols;
    this->matrix.resize(this->row_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            this->matrix[i].resize(this->col_size, 0.0);
        }
    }
};

Matrix::Matrix (vector<vector<double>> data) {
    this->row_size = data.size();
    this->col_size = data[0].size();
    this->matrix = data;
};

Matrix Matrix::dot (const Matrix &mtx) {
    if (this->row_size != mtx.col_size) {
        throw std::runtime_error("Matrixes aren't compatable!");
    }
    Matrix dotProduct(this->row_size, mtx.col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < mtx.col_size; j++) {
            double pool = 0.0;
            for (int k = 0; k < this->col_size; k++){
                pool += this->matrix[i][k] * mtx.matrix[k][j];
            }
            dotProduct.matrix[i][j] = pool;
        }
    }
    return dotProduct;
};

Matrix Matrix::T () {
    Matrix Transposed(this->col_size, this->row_size);

    for (int i = 0; i < this->col_size; i++) {
        for (int j = 0; j < this->row_size; j++) {
            Transposed.matrix[i][j] = this->matrix[j][i];
        }
    }

    return Transposed;
};

Matrix Matrix::sum (const int &axis) {
    Matrix Sum;

    if (axis == 0) {
        Sum = Matrix(1 , this->col_size);
        for (int i = 0; i < this->col_size; i++) {
            for (int j = 0; j < this->row_size; j++) {
                Sum.matrix[0][i] += this->matrix[j][i];
            }
        }
    }
    else if (axis == 1) {
        Sum = Matrix(this->row_size, 1);
        for (int i = 0; i < this->row_size; i++) {
            for (int j = 0; j < this->col_size; j++) {
                Sum.matrix[i][0] += this->matrix[i][j]; 
            }
        }
    }
    else  {
        throw std::runtime_error("Error: There is no " + to_string(axis) + " axis!\n");
    }
    
    return Sum;
};

Matrix Matrix::argmax (const int &axis) {
    Matrix argmaxMx;
    Matrix maxMx = (*this).max(axis);
    if (axis == 0) {
        argmaxMx = Matrix(1 , this->col_size);
        for (int i = 0; i < this->row_size; i++){
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i][j] == maxMx.matrix[0][j])
                    argmaxMx.matrix[0][j] = i;
            }
        }
    }
    else if (axis == 1) {
        argmaxMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i][j] == maxMx.matrix[i][0])
                    argmaxMx.matrix[i][0] = j;
            }
        }
    }
    else {
        throw std::runtime_error("Error: There is no " + to_string(axis) + " axis!\n");
    }
    return argmaxMx;
};

Matrix Matrix::max (const int &axis) {
    Matrix maxMx;
    if (axis == 0) {
        maxMx = Matrix(1 , this->col_size);
        for (int i = 0; i < this->row_size; i++){
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i][j] > maxMx.matrix[0][j])
                    maxMx.matrix[0][j] = this->matrix[i][j];
            }
        }
    }
    else if (axis == 1) {
        maxMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i][j] > maxMx.matrix[i][0])
                    maxMx.matrix[i][0] = this->matrix[i][j];
            }
        }
    }
    else {
        throw std::runtime_error("Error: There is no " + to_string(axis) + " axis!\n");
    }
    return maxMx;
};

tuple<int, int> Matrix::shape () {
    return make_tuple(this->row_size, this->col_size);
};

Matrix Matrix::reshape (int rows, int cols) {
    int size = (this->row_size) * (this->col_size);
    if (rows == -1) {
        rows = size / cols;
    } else if (cols == -1) {
        cols = size / rows;
    }
    if ( (cols * rows) != size) {
        throw std::runtime_error("Cannot reshape, arrays have incompatable size!");
    }

    Matrix reshapedMx = Matrix (rows, cols);
    
    int ind_x = 0, ind_y = 0;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++) {
                reshapedMx.matrix[i][j] = this->matrix[ind_x][ind_y];
                if (ind_y < this->col_size - 1) {
                    ind_y++;
                } else {
                    ind_y = 0;
                    ind_x++;
                }
        }
    }
    
    return reshapedMx;
};

double Matrix::ndim() {
    if (this->row_size != 0 && this->col_size != 0)
        return (this->row_size == 1) || (this->col_size == 1) ? 1.0 : 2.0;
    else
        return 0.0;
};

Matrix Matrix::mean (const int &axis) {
    Matrix meanMx;
    if (axis == 0) {
        meanMx = Matrix(1 , this->col_size);
        for (int i = 0; i < this->row_size; i++){
            for (int j = 0; j < this->col_size; j++){
                meanMx.matrix[0][j] += this->matrix[i][j];
            }
        }
        for (int i = 0; i < this->col_size; i++)
            meanMx.matrix[0][i] = meanMx.matrix[0][i] / this->row_size;

    }
    else if (axis == 1) {
        meanMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            for (int i = 0; i < this->row_size; i++){
                meanMx.matrix[i][0] += this->matrix[i][j];
            }
        }
        for (int i = 0; i < this->row_size; i++)
            meanMx.matrix[i][0] = meanMx.matrix[i][0] / this->col_size;
    }
    else {
        throw std::runtime_error("Error: There is no " + to_string(axis) + " axis!\n");
    }
    return meanMx;
};

Matrix Matrix::log () {
    Matrix LogMX(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            LogMX.matrix[i][j] = std::log(this->matrix[i][j]);
        }
    }
    return LogMX;
};

Matrix Matrix::exp() {
    Matrix ExpMX(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            ExpMX.matrix[i][j] =  std::exp(this->matrix[i][j]);
        }
    }
    return ExpMX;
};

///TODO: implement the broadcast function
tuple<Matrix> broadcast (Matrix &Mx, tuple<int, int> &shape) {
    int width = std::get<1>(Mx.shape());
    int height = std::get<0>(Mx.shape());
    int desire_w = std::get<1>(shape);
    int desire_h = std::get<0>(shape);

    if (width == desire_w || width == 1){

    }
    else {
        throw std::runtime_error("Error: Matrix (" + to_string(width) + ", " + to_string(height) +
            ") cannot be broadcasted to shape (" + to_string(desire_w) + ", " + to_string(desire_h) + ")!\n");
    }
    if (width == desire_w || width == 1){

    }
    else {
        throw std::runtime_error("Error: Matrix (" + to_string(width) + ", " + to_string(height) +
            ") cannot be broadcasted to shape (" + to_string(desire_w) + ", " + to_string(desire_h) + ")!\n");
    }
};

///TODO: implement operations over matrixes
///DEPENDED: broadcasting function
//Matrix operator + (Matrix &) {};
//Matrix operator - (Matrix &) {};
Matrix Matrix::operator * (Matrix &mtx) {
    //broadcast((*this), mtx);
    Matrix MultyMx;
    return MultyMx;
};

Matrix Matrix::operator * (double val) {
    Matrix MultyMx(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            MultyMx.matrix[i][j] = this->matrix[i][j] * val;
        }
    }
    return MultyMx;
};

Matrix Matrix::operator + (double val) {
    Matrix AddMx(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            AddMx.matrix[i][j] = this->matrix[i][j] + val;
        }
    }
    return AddMx;
};

Matrix Matrix::operator - (double val) {
    Matrix DiffMx = Matrix(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            DiffMx.matrix[i][j] = this->matrix[i][j] - val;
        }
    }
    return DiffMx;
};

Matrix Matrix::operator ^ (const double &deg) {
    Matrix PowMx;
    PowMx = (*this).T();
    for (int i = 1; i < deg; i++){
        //if (i % 2 != 0){
        //    PowMx = PowMx.dot((*this).T());
        //}
        //else {
        //    PowMx = PowMx.dot(*this);
        //}
        ///TODO: fix the issues.
    }
    return PowMx;
};

Matrix& Matrix::operator = (const Matrix & mtx) {
    this->row_size = mtx.row_size;
    this->col_size = mtx.col_size;
    this->matrix = mtx.matrix;
    return (*this);
};

Matrix& Matrix::operator = (const double & val) {
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            this->matrix[i][j] = this->matrix[i][j] * val;
        }
    }
    return (*this);
};

double& Matrix::operator () (const int &row, const int &col) {
    return this->matrix[row][col];
};

//Pretty print function that outstreams 2-dim matrixes
//directly to std::ofstream.
void Matrix::print () {
    cout << "[";
    for (int i = 0; i < row_size; i++) {
        (i == 0) ? cout << "" : cout << " ";
        cout << " [";
        for (int j = 0; j < col_size; j++) {
            cout << " " << matrix[i][j];
        }
        cout << " ]";
        (i != row_size - 1) ? cout << "\n" : cout << "";
    }
    cout << " ]\n";
};