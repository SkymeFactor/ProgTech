#pragma once
#include <fstream>
#include <vector>
#include "MatrixLib/Matrix.hpp"

using std::vector;
using std::pair;

class DataLoader {
private:
    std::ifstream fin;
public:
    pair<vector<vector<double>>, vector<double>> load_dataset(const char*, pair<int, int>);
    static pair<Matrix, Matrix> load_as_matrix (vector<vector<double>> &, vector<double> &, vector<int> &);
    static void prepare_dataset (vector<vector<double>> &, vector<vector<double>> &);
};