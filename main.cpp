#include <iostream>
#include "Matrix.hpp"

int main (int argc, char ** argv) {
    //Test area of the Numpy ("Matrix" in my notation) functions.
    Matrix m = Matrix(3, 2);
    std::cout << "Matrix:\n";
    m.print();
    std::cout << "Shape: " << std::get<0>(m.shape()) << " " << std::get<1>(m.shape()) << "\n";
    std::cout << "Transposed:\n";
    m = m.T();
    m.print();
    std::cout << "Shape: " << std::get<0>(m.shape()) << " " << std::get<1>(m.shape()) << "\n";
    Matrix a = m - 4.0;
    std::cout << "Matrix minus scalar:\n";
    a.print();
    std::cout << "Summarized by axis 0:\n";
    a.sum(0).print();
    std::cout << "Matrix plus scalar:\n";
    a = a + 2.0;
    a.print();
    std::cout << "Matrix multiplyed by scalar:\n";
    (a * 0.03).print();
    std::cout << "Dot product:\n";
    (a.dot(a.T()+6)).print();
    std::cout << "Power of matrix:\n";
    (a ^ 3.0).print();
    std::cout << "Loaded matrix:\n";
    vector<vector<double>> v;
    v.push_back(vector<double>{2, 5});
    v.push_back(vector<double>{6, 1});
    v.push_back(vector<double>{3, 7});
    Matrix b = Matrix(v);
    b.print();
    std::cout << "Argmax by axis 0:\n";
    (b.argmax(0)).print();
    std::cout << "Mean matrix:\n";
    (b.mean(1)).print();
    std::cout << "Log matrix:\n";
    (b.log()).print();
    std::cout << "Reshaped matrix:\n";
    (b.reshape(1, -1)).print();
    //b.T().print();
    
    return 0;
}