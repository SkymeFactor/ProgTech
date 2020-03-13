#pragma once

#include <vector>
#include <tuple>

using std::vector;
using std::tuple;

/***********************************************************
 * Class Matrix is a class that holds the values inside
 * in a form of 2-dim array. It was developed to provide
 * a better experience while working with matrixes in terms
 * of NN's or ML. Most of it's functionality was inherited
 * on a conceptual level from the NumPy pachage.
 * --------------------------------------------------------
 * Known issues:
 *    -Unfortunately, at the moment it only supports working
 *    with two dimensional arrays.
 *    -Doesn't support slicing.
 *    -Doesn't contain lots of crucial functions.
 *    -All functions are experiencing lack of key arguments.
 *    -Getting a value by it's index should be done by using
 *    the round brackets instead of a squared ones.
 * --------------------------------------------------------
 * Last changes 13 mar 2020 by Skyme Factor.
 **********************************************************/
class Matrix {
private :
    int row_size; //rows i.e. x-axis
    int col_size; //columns i.e. y-axis
    vector<vector<double>> matrix; //matrix itself

public:
    Matrix();
    Matrix(int, int);
    Matrix(vector<vector<double>>);
    Matrix dot (const Matrix &);
    Matrix T ();
    Matrix sum (const int &);
    Matrix argmax (const int &);
    Matrix max (const int &);
    tuple<int, int> shape ();
    Matrix reshape (int, int);
    double ndim ();
    Matrix mean (const int &);
    Matrix log ();
    Matrix exp ();
    friend Matrix broadcast (Matrix &, tuple<int, int>);
    Matrix operator * (Matrix &);
    Matrix operator * (double);
    Matrix operator + (Matrix &);
    Matrix operator + (double);
    Matrix operator - (Matrix &);
    Matrix operator - (double);
    Matrix operator ^ (const double &); //Matrixes-powering
    Matrix& operator = (const Matrix &);
    Matrix& operator = (const double &);
    double& operator () (const int &, const int &);
    void print();
};