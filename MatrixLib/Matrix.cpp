#include "Matrix.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <math.h>
#include <omp.h>


Matrix::Matrix (int rows, int cols) {
    this->row_size = rows;
    this->col_size = cols;
    this->matrix.resize(this->row_size * this->col_size);
};

Matrix::Matrix (vector<vector<double>> data) {
    this->row_size = data.size();
    this->col_size = data[0].size();
    for (auto it_1 : data){
        for (auto it_2 : it_1)
            this->matrix.push_back(it_2);
    }
};

Matrix& Matrix::fill_ones () {
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        this->matrix[i] = 1;
    }
    return (*this);
}

Matrix& Matrix::fill_zeros () {
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        this->matrix[i] = 0;
    }
    return (*this);
}

Matrix& Matrix::fill_rand () {
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++){
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d(0, 1);

        this->matrix[i] = d(gen);
    }
    return (*this);
}

Matrix Matrix::dot (const Matrix &mtx) {
    if (this->col_size != mtx.row_size) {
        throw std::runtime_error("Matrices aren't compatible!");
    }
    Matrix mtx_transposed = mtx.T();
    Matrix dotProduct(this->row_size, mtx.col_size);

    #pragma omp parallel for
    for (int i = 0; i < this->row_size; i++) {
        int i_offset = i * this->col_size;
        for (int j = 0; j < mtx.col_size; j++) {
            int j_offset = j * mtx_transposed.col_size;
            double product_sum = 0.0;
            for (int k = 0; k < this->col_size; k++){
                dotProduct.matrix[i * dotProduct.col_size + j] += this->matrix[i_offset + k] * mtx_transposed.matrix[j_offset + k];
            }
        }
    }
    return dotProduct;
};

Matrix Matrix::T () const {
    Matrix Transposed(this->col_size, this->row_size);
    #pragma omp parallel for
    for (int i = 0; i < this->col_size; i++) {
        int i_offset = i * this->row_size;
        for (int j = 0; j < this->row_size; j++) {
            Transposed.matrix[i_offset + j] = this->matrix[j * this->col_size + i];
        }
    }

    return Transposed;
};

Matrix Matrix::sum (const int &axis) {
    Matrix Sum;

    if (axis == 0) {
        Sum = Matrix(1 , this->col_size);
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++) {
            double col_sum = 0.0;
            for (int i = 0; i < this->row_size; i++) {
                Sum.matrix[j] += this->matrix[i * this->col_size + j];
            }
        }
    }
    else if (axis == 1) {
        Sum = Matrix(this->row_size, 1);
        #pragma omp parallel for
        for (int i = 0; i < this->row_size; i++) {
            int i_offset = i * this->col_size;
            double row_sum = 0.0;
            #pragma omp parallel for reduction(+: row_sum)
            for (int j = 0; j < this->col_size; j++) {
                Sum.matrix[i] += this->matrix[i_offset + j];
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
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++){
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i * this->col_size + j] == maxMx.matrix[j])
                    argmaxMx.matrix[j] = i;
            }
        }
    }
    else if (axis == 1) {
        argmaxMx= Matrix(this->row_size, 1);
        #pragma omp prarallel for
        for (int i = 0; i < this->row_size; i++){
            int i_offset = i * this->col_size;
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i_offset + j] == maxMx.matrix[i])
                    argmaxMx.matrix[i] = j;
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
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++){
            for (int i = 0; i < this->row_size; i++){
                if (this->matrix[i * this->col_size + j] > maxMx.matrix[j])
                    maxMx.matrix[j] = this->matrix[i * this->col_size + j];
            }
        }
    }
    else if (axis == 1) {
        maxMx= Matrix(this->row_size, 1);
        #pragma omp parallel for
        for (int i = 0; i < this->row_size; i++){
            int i_offset = i * this->col_size;
            for (int j = 0; j < this->col_size; j++){
                if (this->matrix[i_offset + j] > maxMx.matrix[i])
                    maxMx.matrix[i] = this->matrix[i_offset + j];
            }
        }
    }
    else {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    return maxMx;
};

tuple<int, int> Matrix::shape () {
    return std::tuple<int, int>(this->row_size, this->col_size);
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

    Matrix reshapedMx = (*this);

    reshapedMx.row_size = rows;
    reshapedMx.col_size = cols;

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
        #pragma omp parallel for
        for (int j = 0; j < this->col_size; j++){
            double col_sum = 0.0;
            #pragma omp parallel for reduction(+: col_sum)
            for (int i = 0; i < this->row_size; i++){
                col_sum += this->matrix[i * this->col_size + j];
            }
            meanMx.matrix[j] = col_sum;
        }
        #pragma omp parallel for
        for (int j = 0; j < meanMx.col_size; j++)
            meanMx.matrix[j] = meanMx.matrix[j] / this->row_size;

    }
    else if (axis == 1) {
        meanMx= Matrix(this->row_size, 1);
        #pragma omp parallel for
        for (int i = 0; i < this->row_size; i++){
            int i_offset = i * this->col_size;
            double row_sum = 0.0;
            #pragma omp parallel for reduction(+: row_sum)
            for (int j = 0; j < this->col_size; j++){
                row_sum += this->matrix[i_offset + j];
            }
            meanMx.matrix[i] = row_sum;
        }
        #pragma omp parallel for
        for (int i = 0; i < meanMx.col_size; i++)
            meanMx.matrix[i] = meanMx.matrix[i] / this->col_size;
    }
    else {
        throw std::runtime_error("Error: There is no " + std::to_string(axis) + " axis!\n");
    }
    return meanMx;
};

Matrix Matrix::log () {
    Matrix LogMx(this->row_size, this->col_size);
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        LogMx.matrix[i] = std::log(this->matrix[i]);
    }
    return LogMx;
};

Matrix Matrix::exp() {
    Matrix ExpMx(this->shape());
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        ExpMx.matrix[i] =  std::exp(this->matrix[i]);
    }
    return ExpMx;
};

Matrix Matrix::sqrt() {
    Matrix SqrtMx(this->row_size, this->col_size);
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        SqrtMx.matrix[i] = std::sqrt(this->matrix[i]);
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
        BroadcastMx.matrix.resize(BroadcastMx.row_size * BroadcastMx.col_size);
        #pragma omp parallel for
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < desire_c; j++){
                BroadcastMx.matrix[i * BroadcastMx.col_size + j] = this->matrix[i];
            }
    }
    else if (cols != desire_c) {
        throw std::runtime_error("Error: Matrix (" + std::to_string(rows) + ", " + std::to_string(cols) +
            ") cannot be broadcasted to shape (" + std::to_string(desire_r) + ", " + std::to_string(desire_c) + ")!\n");
    }

    if (rows == 1){
        BroadcastMx.row_size = desire_r;
        BroadcastMx.matrix.resize(BroadcastMx.row_size * BroadcastMx.col_size);
        #pragma omp parallel for
        for (int i = 0; i < desire_r; i++)
            for (int j = 0; j < BroadcastMx.col_size; j++){
                BroadcastMx.matrix[i * BroadcastMx.col_size + j] = BroadcastMx.matrix[j];
            }
    }
    else if (rows != desire_r) {
        throw std::runtime_error("Error: Matrix (" + std::to_string(rows) + ", " + std::to_string(cols) +
            ") cannot be broadcasted to shape (" + std::to_string(desire_r) + ", " + std::to_string(desire_c) + ")!\n");
    }

    return BroadcastMx;
};

tuple<int, int> Matrix::broadcast_shape (tuple<int, int> l_shape, tuple<int, int> r_shape) {
    int rows = std::max(std::get<0>(l_shape), std::get<0>(r_shape));
    int cols = std::max(std::get<1>(l_shape), std::get<1>(r_shape));

    return std::tuple<int, int>(rows, cols);
};

Matrix Matrix::operator + (Matrix &mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix AddMx(std::get<0>(shape), std::get<1>(shape));

    #pragma omp parallel for
    for (int i = 0; i < l.row_size * l.col_size; i++)
        AddMx.matrix[i] = l.matrix[i] + r.matrix[i];

    return AddMx;
};

Matrix Matrix::operator - (Matrix & mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix SubMx(std::get<0>(shape), std::get<1>(shape));
    
    #pragma omp parallel for
    for (int i = 0; i < l.row_size * l.col_size; i++)
        SubMx.matrix[i] = l.matrix[i] - r.matrix[i];

    return SubMx;
};

Matrix Matrix::operator * (Matrix mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = mtx.broadcast(shape);
    Matrix MultyMx(std::get<0>(shape), std::get<1>(shape));
    #pragma omp parallel for
    for (int i = 0; i < l.row_size * l.col_size; i++){
        MultyMx.matrix[i] = l.matrix[i] * r.matrix[i];
    }
    

    return MultyMx;
};

Matrix Matrix::operator * (double val) {
    Matrix MultyMx(this->shape());
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        MultyMx.matrix[i] = this->matrix[i] * val;
    }
    return MultyMx;
};

Matrix Matrix::operator + (double val) {
    Matrix AddMx(this->shape());
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        AddMx.matrix[i] = this->matrix[i] + val;
    }
    return AddMx;
};

Matrix Matrix::operator - (double val) {
    Matrix DiffMx(this->shape());
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        DiffMx.matrix[i] = this->matrix[i] - val;
    }
    return DiffMx;
};

Matrix Matrix::operator - () {
    Matrix MinusMx(this->shape());

    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        MinusMx.matrix[i] = - this->matrix[i];
    }
    return MinusMx;
};

Matrix Matrix::operator / (Matrix &&mtx) {
    auto shape = broadcast_shape((*this).shape(), mtx.shape());
    Matrix l = (*this).broadcast(shape);
    Matrix r = std::forward<Matrix>(mtx.broadcast(shape));
    Matrix DivMx(shape);

    #pragma omp parallel for
    for (int i = 0; i < l.row_size * l.col_size; i++){
        DivMx.matrix[i] = l.matrix[i] / r.matrix[i];
    }
    
    return DivMx;
}

Matrix Matrix::operator / (const double &val) {
    Matrix DivMx(this->shape());
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        DivMx.matrix[i] = this->matrix[i] / val;
    }
    return DivMx;
}

Matrix Matrix::operator ^ (const double &deg) {
    Matrix PowMx;
    PowMx = (*this);
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++){
        PowMx.matrix[i] = std::pow(PowMx.matrix[i], deg);
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
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
        this->matrix[i] = val;
    }
    return (*this);
};

bool Matrix::operator == (const Matrix & mtx) {
    if (this->row_size != mtx.row_size || this->col_size != mtx.col_size) {
        return false;
    }else {
        bool result = true;
        #pragma omp parallel for reduction(&: result)
        for (int i = 0; i < this->row_size * this->col_size; i++) {
            if ( this->matrix[i] != mtx.matrix[i])
                result = false;
        }
        return result;
    }
};

Matrix Matrix::operator > (double val) {
    Matrix GtMx(this->shape());
    
    #pragma omp parallel for
    for (int i = 0; i < this->row_size * this->col_size; i++) {
            if (this->matrix[i] > val)
                GtMx.matrix[i] = 1;
            else
                GtMx.matrix[i] = 0;
    }
    return GtMx;
}

double& Matrix::operator () (const int &row, const int &col) {
    if (row > this->row_size - 1 || col > this->col_size - 1)
        throw std::runtime_error("Requested index is out of matrix's bounds");
    else {
        return this->matrix[row * this->col_size + col];
    }
};

//Pretty print function that outstreams 2-dim matrices
//directly to std::ofstream.
void Matrix::print (bool np_insert) {
    std::cout << "[";
    for (int i = 0; i < this->row_size; i++) {
        (i == 0) ? std::cout << "" : std::cout << " ";
        std::cout << " [";
        for (int j = 0; j < this->col_size; j++) {
            std::cout << " " << this->matrix[i * this->col_size + j];
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