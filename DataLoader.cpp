#include <fstream>
#include <iostream>
#include <chrono>
#include "DataLoader.hpp"


pair<vector<vector<double>>, vector<double>> DataLoader::load_dataset(const char* filename, pair<int, int> size) {
    // Open the file
    fin.open(filename, std::ios::binary);
    // Form the dataset of size of images x image_size
    int images = size.first, image_size = size.second;
    // Create the matrices
    vector<vector<double>> dataset;
    dataset.resize(images, vector<double>(image_size));
    vector<double> labels(images);
    //Matrix labels = Matrix(1, images);
    // Temporary variables, ifstream.read(...) requires char* whereas we
    // need colors to be unsigned char within the range from 0 to 255
    char *temp = new char[image_size];
    unsigned char *u_temp;

    for (int i = 0; i < images; i++){
        // Reading the image
        fin.read(temp, image_size);
        u_temp = reinterpret_cast<unsigned char*>(temp);
        #pragma omp parallel for shared(i, u_temp, dataset)
        for (int j = 0; j < image_size; j++){
            dataset[i][j] = (double)u_temp[j];
        }
        
        // Reading the label
        fin.read(temp, 1);
        u_temp = reinterpret_cast<unsigned char*>(temp);
        labels[i] = (double)u_temp[0];
    }
    fin.close();

    return pair<vector<vector<double>>, vector<double>>(dataset, labels);
}

pair<Matrix, Matrix> DataLoader::load_as_matrix (vector<vector<double>> &X, vector<double> &y, vector<int> &indices) {
    vector<vector<double>> result_images(indices.size());
    vector<vector<double>> result_labels(indices.size());

    for (int i = 0; i < (int)indices.size(); i++){
        result_images[i] = X[indices[i]];
        result_labels[i] = vector<double>{y[indices[i]]};
    }

    return pair<Matrix, Matrix>(Matrix(result_images), Matrix(result_labels));
}

void DataLoader::prepare_dataset (vector<vector<double>> &Train, vector<vector<double>> &Test) {
    
    auto t1 = std::chrono::high_resolution_clock::now();
    // Cast the Train colors to the 0.0..1.0 ratio
    for (int i = 0; i < (int)Train.size(); i++){
        int size = (int)Train[i].size();
        #pragma omp parallel for shared(i, size, Train)
        for (int j = 0; j < size; j++)
            Train[i][j] /= 255.0;
    }

    // Cast the Test colors to the 0.0..1.0 ratio
    for (int i = 0; i < (int)Test.size(); i++){
        int size = (int)Test[i].size();
        #pragma omp parallel for shared(i, size, Test)
        for (int j = 0; j < size; j++)
            Test[i][j] /= 255.0;
    }

    // Compute mean value
    double mean = 0.0;
    for (auto it_1 : Train) {
        int size = (int)it_1.size();
        #pragma omp parallel for shared(it_1, size) reduction(+: mean)
        for (int i = 0; i < size; i++){
            mean += it_1[i];
        }
    }

    mean /= (double)(Train.size() * Train[0].size());
    
    // Normalize Train
    for (int i = 0; i < (int)Train.size(); i++){
        int size = (int)Train[i].size();
        #pragma omp parallel for shared(i, size, Train)
        for (int j = 0; j < size; j++)
            Train[i][j] -= mean;
    }

    // Normalize Test
    for (int i = 0; i < (int)Test.size(); i++){
        int size = (int)Test[i].size();
        #pragma omp parallel for shared(i, size, Train)
        for (int j = 0; j < size; j++)
            Test[i][j] -= mean;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Dataset prepared for " << dur / 1000.0 << " milliseconds\n";
}
