#include "Matrix.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <math.h>
#include <omp.h>

Matrix::Matrix(){
    this->row_size = 1;
    this->col_size = 1;
    this->matrix.resize(this->row_size);
    this->matrix[0].resize(this->col_size);
};

Matrix::Matrix (int rows, int cols) {
    this->row_size = rows;
    this->col_size = cols;
    this->matrix.resize(this->row_size);
    #pragma omp parallel for
    for (int i = 0; i < this->row_size; i++) {
        this->matrix[i].resize(this->col_size);
    }
};

Matrix::Matrix (vector<vector<double>> data) {
    this->row_size = data.size();
    this->col_size = data[0].size();
    this->matrix = data;
};

Matrix& Matrix::fill_ones () {
    for (int i = 0; i < this->row_size; i++)
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++)
            this->matrix[i][j] = 1;
    return (*this);
}

Matrix& Matrix::fill_zeros () {
    for (int i = 0; i < this->row_size; i++)
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++)
            this->matrix[i][j] = 0;
    return (*this);
}

Matrix& Matrix::fill_rand () {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d(0, 1);
    for (int i = 0; i < this->row_size; i++)
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++)
            this->matrix[i][j] = d(gen);
    return (*this);
}

Matrix Matrix::dot (const Matrix &mtx) {
    if (this->col_size != mtx.row_size) {
        throw std::runtime_error("Matrices aren't compatible!");
    }
    Matrix dotProduct(this->row_size, mtx.col_size);
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < mtx.col_size; j++) {
            for (int k = 0; k < this->col_size; k++){
                dotProduct.matrix[i][j] += this->matrix[i][k] * mtx.matrix[k][j];
            }
        }
    }
    return dotProduct;
};

Matrix Matrix::T () {
    Matrix Transposed(this->col_size, this->row_size);

    for (int i = 0; i < this->col_size; i++) {
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (int i = 0; i < this->col_size; i++) {
            for (int j = 0; j < this->row_size; j++) {
                Sum.matrix[0][i] += this->matrix[j][i];
            }
        }
    }
    else if (axis == 1) {
        Sum = Matrix(this->row_size, 1);
        #pragma omp parallel for
        for (int i = 0; i < this->row_size; i++) {
            for (int j = 0; j < this->col_size; j++) {
                Sum.matrix[i][0] += this->matrix[i][j]; 
            }
        }
    }
    else  {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    
    return Sum;
};

Matrix Matrix::argmax (const int &axis) {
    Matrix argmaxMx;
    Matrix maxMx = (*this).max(axis);
    if (axis == 0) {
        argmaxMx = Matrix(1 , this->col_size);
        for (int i = 0; i < this->row_size; i++){
            #pragma omp parallel for
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i][j] == maxMx.matrix[0][j])
                    argmaxMx.matrix[0][j] = i;
            }
        }
    }
    else if (axis == 1) {
        argmaxMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            #pragma omp parallel for
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i][j] == maxMx.matrix[i][0])
                    argmaxMx.matrix[i][0] = j;
            }
        }
    }
    else {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    return argmaxMx;
};

Matrix Matrix::max (const int &axis) {
    Matrix maxMx;
    if (axis == 0) {
        maxMx = Matrix(1 , this->col_size);
        for (int i = 0; i < this->row_size; i++){
            #pragma omp parallel for
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i][j] > maxMx.matrix[0][j])
                    maxMx.matrix[0][j] = this->matrix[i][j];
            }
        }
    }
    else if (axis == 1) {
        maxMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            #pragma omp parallel for
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i][j] > maxMx.matrix[i][0])
                    maxMx.matrix[i][0] = this->matrix[i][j];
            }
        }
    }
    else {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    return maxMx;
};

tuple<int, int> Matrix::shape () {
    return std::tuple(this->row_size, this->col_size);
};

Matrix Matrix::reshape (int rows, int cols) {
    int size = (this->row_size) * (this->col_size);
    if (rows == -1) {
        rows = size / cols;
    } else if (cols == -1) {
        cols = size / rows;
    }
    if ( (cols * rows) != size) {
        throw std::runtime_error("Cannot reshape, arrays have incompatible size!");
    }

    Matrix reshapedMx = Matrix (rows, cols);
    
    int ind_x = 0, ind_y = 0;
    for (int i = 0; i < rows; i++){
        #pragma omp parallel for
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
            #pragma omp parallel for
            for (int j = 0; j < this->col_size; j++){
                meanMx.matrix[0][j] += this->matrix[i][j];
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < this->col_size; i++)
            meanMx.matrix[0][i] = meanMx.matrix[0][i] / this->row_size;

    }
    else if (axis == 1) {
        meanMx= Matrix(this->row_size, 1);
        for (int j = 0; j < this->col_size; j++){
            #pragma omp parallel for
            for (int i = 0; i < this->row_size; i++){
                meanMx.matrix[i][0] += this->matrix[i][j];
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < this->row_size; i++)
            meanMx.matrix[i][0] = meanMx.matrix[i][0] / this->col_size;
    }
    else {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    return meanMx;
};

Matrix Matrix::log () {
    Matrix LogMx(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            LogMx.matrix[i][j] = std::log(this->matrix[i][j]);
        }
    }
    return LogMx;
};

Matrix Matrix::exp() {
    Matrix ExpMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            ExpMx.matrix[i][j] =  std::exp(this->matrix[i][j]);
        }
    }
    return ExpMx;
};

Matrix Matrix::sqrt() {
    Matrix SqrtMx(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            SqrtMx.matrix[i][j] =  std::sqrt(this->matrix[i][j]);
        }
    }
    return SqrtMx;
}


Matrix Matrix::broadcast (tuple<int, int> shape) {
    int cols = this->col_size;
    int rows = this->row_size;
    int desire_c = std::get<1>(shape);
    int desire_r = std::get<0>(shape);
    Matrix BroadcastMx = (*this);

    if (cols == 1){
        BroadcastMx.col_size = desire_c;
        for (int j = 0; j < rows; j++)
            for (int i = 1; i < desire_c; i++){
                BroadcastMx.matrix[j].push_back(BroadcastMx.matrix[j][0]);
            }
    }
    else if (cols != desire_c) {
        throw std::runtime_error("Error: Matrix (" + std::to_string(cols) + ", " + std::to_string(rows) +
            ") cannot be broadcasted to shape (" + std::to_string(desire_c) + ", " + std::to_string(desire_r) + ")!\n");
    }

    if (rows == 1){
        BroadcastMx.row_size = desire_r;
        for (int i = 1; i < desire_r; i++){
            BroadcastMx.matrix.push_back(BroadcastMx.matrix[0]);
        }
    }
    else if (rows != desire_r) {
        throw std::runtime_error("Error: Matrix (" + std::to_string(cols) + ", " + std::to_string(rows) +
            ") cannot be broadcasted to shape (" + std::to_string(desire_c) + ", " + std::to_string(desire_r) + ")!\n");
    }

    return BroadcastMx;
};

tuple<int, int> Matrix::broadcast_shape (tuple<int, int> l_shape, tuple<int, int> r_shape) {
    int rows = std::max(std::get<0>(l_shape), std::get<0>(r_shape));
    int cols = std::max(std::get<1>(l_shape), std::get<1>(r_shape));

    return std::tuple(rows, cols);
};

Matrix Matrix::operator + (Matrix &mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix AddMx(std::get<0>(shape), std::get<1>(shape));

    for (int i = 0; i < l.row_size; i++)
        #pragma omp parallel for
        for (int j = 0; j < l.col_size; j++){
            AddMx.matrix[i][j] = l.matrix[i][j] + r.matrix[i][j];
        }

    return AddMx;
};

Matrix Matrix::operator - (Matrix & mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix SubMx(std::get<0>(shape), std::get<1>(shape));

    for (int i = 0; i < l.row_size; i++)
        #pragma omp parallel for
        for (int j = 0; j < l.col_size; j++){
            SubMx.matrix[i][j] = l.matrix[i][j] - r.matrix[i][j];
        }

    return SubMx;
};

Matrix Matrix::operator * (Matrix mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix MultyMx(std::get<0>(shape), std::get<1>(shape));

    for (int i = 0; i < l.row_size; i++){
        #pragma omp parallel for
        for (int j = 0; j < l.col_size; j++){
            MultyMx.matrix[i][j] = l.matrix[i][j] * r.matrix[i][j];
        }
    }
    

    return MultyMx;
};

Matrix Matrix::operator * (double val) {
    Matrix MultyMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            MultyMx.matrix[i][j] = this->matrix[i][j] * val;
        }
    }
    return MultyMx;
};

Matrix Matrix::operator + (double val) {
    Matrix AddMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            AddMx.matrix[i][j] = this->matrix[i][j] + val;
        }
    }
    return AddMx;
};

Matrix Matrix::operator - (double val) {
    Matrix DiffMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            DiffMx.matrix[i][j] = this->matrix[i][j] - val;
        }
    }
    return DiffMx;
};

Matrix Matrix::operator - () {
    Matrix MinusMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            MinusMx.matrix[i][j] = - this->matrix[i][j];
        }
    }
    return MinusMx;
};

Matrix Matrix::operator / (Matrix &&mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = std::forward<Matrix>(mtx.broadcast(shape));
    Matrix DivMx(shape);

    for (int i = 0; i < l.row_size; i++){
        #pragma omp parallel for
        for (int j = 0; j < l.col_size; j++){
            DivMx.matrix[i][j] = l.matrix[i][j] / r.matrix[i][j];
        }
    }
    
    return DivMx;
}

Matrix Matrix::operator / (const double &val) {
    Matrix DivMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            DivMx.matrix[i][j] = this->matrix[i][j] / val;
        }
    }
    return DivMx;
}

Matrix Matrix::operator ^ (const double &deg) {
    Matrix PowMx;
    PowMx = (*this);
    for (int i = 0; i < PowMx.row_size; i++){
        #pragma omp parallel for
        for (int j = 0; j < PowMx.col_size; j++){
            PowMx.matrix[i][j] = std::pow(PowMx.matrix[i][j], deg);
        }
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
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            this->matrix[i][j] = val;
        }
    }
    return (*this);
};

Matrix Matrix::operator > (double val) {
    Matrix GtMx(this->shape());
    for (int i = 0; i < this->row_size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {  
            if (this->matrix[i][j] > val)
                GtMx.matrix[i][j] = 1;
            else
                GtMx.matrix[i][j] = 0;
        }
    }
    return GtMx;
}

double& Matrix::operator () (const int &row, const int &col) {
    return this->matrix[row][col];
};

//Pretty print function that outstreams 2-dim matrices
//directly to std::ofstream.
void Matrix::print (bool np_insert) {
    std::cout << "[";
    for (int i = 0; i < this->row_size; i++) {
        (i == 0) ? std::cout << "" : std::cout << " ";
        std::cout << " [";
        for (int j = 0; j < this->col_size; j++) {
            std::cout << " " << this->matrix[i][j];
            if (j < this->col_size - 1 && np_insert){
                std::cout << ",";
            }
        }
        std::cout << " ]";
        if (i < this->row_size - 1 && np_insert){
            std::cout << ",";
        }
        (i != this->row_size - 1) ? std::cout << "\n" : std::cout << "";
    }
    std::cout << " ]\n";
};