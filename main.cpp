#include <iostream>
#include "Matrix.hpp"
#include "NeuralNet.hpp"

int main (int argc, char * argv[]) {
    
    // Test area of the entire functional, use test as an argument to see it

    if (argc > 1){
        if ( std::string(argv[1]).compare(std::string("test")) == 0 ) {
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
            (a.dot(a.T() + 6)).print();
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
            std::cout << "Matrix mean:\n";
            (b.mean(0).mean(1)).print();
            std::cout << "Log matrix:\n";
            (b.log()).print();
            std::cout << "Reshaped matrix:\n";
            b = b.reshape(1, -1);
            b.print();
            std::cout << "Broadcasted matrix [1x1]:\n";
            a = (Matrix(1, 1) + 1).broadcast(std::make_tuple(2, 1));
            a.print();
            std::cout << "Elementwise addiction with broadcasting:\n";
            (a + b).print();
            std::cout << "Elementwise multiplication with broadcasting:\n";
            a = Matrix(vector<vector<double>>{{0, 1, 2}});
            b = Matrix(vector<vector<double>>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
            (a * b).print();
            
            std::cout << "Operator greater:\n";
            (b > 4.0).print();
            
            std::cout << "ReLU function:\n";
            a = Matrix(vector<vector<double>>{{4, -1, -2}, {-3, 4, 5}, {6, -7, 8}});
            nn::ReLULayer layer;
            layer.forward(a).print();

            std::cout << "FCLayer:\n";
            Matrix x( vector<vector<double>>{ { 1, -2, 3 }, { -1, 2, 0.1 } } );
            nn::FCLayer fc_layer(3, 4);
            fc_layer.forward(x);
            x = Matrix(vector<vector<double>>{ { 1, -2, 3, 4 }, { -1, 2, 0.1, 1 } });
            fc_layer.backward(x).print();

            std::cout << "Optimizer check:\n";
            Matrix d_w( vector<vector<double>>{ { -0.980758, -0.980758, -0.980758, -0.980758 }, { 0.415141, 0.0565105, 0.415141, 0.415141 } } );
            nn::AdamOptim<double> optim;
            optim.update(x, d_w, 0.9).print();

            std::cout << "Softmax check:\n";
            Matrix preds(vector<vector<double>>{ { 0.1, 0.2, 0.3, 0.4 }, { 0.4, 0.3, 0.2, 0.1 }});
            Matrix gt_ind(1, 1);
            gt_ind = Matrix(vector<vector<double>>{ { 3, 1 }});
            std::cout << "ce_loss: " << nn::SoftmaxLayer::ce_loss(preds, gt_ind) << "\nsoftmax:\n";
            nn::SoftmaxLayer::softmax(preds).print();
            std::cout << "softmax_with_ce_loss loss: " << \
                nn::SoftmaxLayer::softmax_with_ce_loss(preds, gt_ind).first << "\nsoftmax_with_ce_loss grad:\n";
            nn::SoftmaxLayer::softmax_with_ce_loss(preds, gt_ind).second.print();
            std::cout << "l2_reg loss: \n" << nn::SoftmaxLayer::l2_reg(preds, 0.9).first << "\nl2_reg grad:\n";
            nn::SoftmaxLayer::l2_reg(preds, 0.9).second.print();

            std::cout << "Actual nn practice:\n";
            nn::Model model(8, 2, 4, 1e-2);
            x = Matrix(vector<vector<double>>{ { 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1 }});
            Matrix y(1, 1);
            y = 1;
            for (int i = 0; i < 3; i++)
                std::cout << "Loss: " << model.feed_forward(x, y) << "\n";
        }
    } else {
    // Real project comes here


    }

    return 0;
}