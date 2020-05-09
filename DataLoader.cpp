#include <fstream>
#include <tuple>
#include <omp.h>
#include "NeuralNet.hpp"
#include "Matrix.hpp"


class DataLoader {
private:
    std::ifstream fin;
    std::ofstream fout;
public:
    std::pair<Matrix, Matrix> load_dataset(const char* filename, std::tuple<int, int> size) {
        // Open the file
        fin.open(filename, std::ios::binary);
        int rows = std::get<0>(size), cols = std::get<1>(size);
        // Create the matrices
        Matrix dataset = Matrix(rows, cols);
        Matrix labels = Matrix(1, rows);
        // Temporary variables, ifstream.read(...) requires char* whereas we
        // need colors to be undigned char from 0 to 255
        char *temp = new char[cols];
        unsigned char *u_temp;

        for (int i = 0; i < rows; i++){
            // Reading the image
            fin.read(temp, cols);
            u_temp = reinterpret_cast<unsigned char*>(temp);
            #pragma omp parallel for
            for (int j = 0; j < cols; j++){
                dataset(i, j) = (double)u_temp[j];
            }
            
            // Reading the label
            fin.read(temp, 1);
            u_temp = reinterpret_cast<unsigned char*>(temp);
            labels(0, i) = (double)u_temp[0];
        }
        fin.close();

        return std::pair<Matrix, Matrix>(dataset, labels);
    }
};

int main() {
    DataLoader dl;

    dl.load_dataset("data/test_32x32.dat", std::tuple<int, int>(26032, 3072));
    dl.load_dataset("data/train.dat", std::tuple<int, int>(73257, 3072));

    return 0;
}