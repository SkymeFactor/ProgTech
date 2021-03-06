#include <iostream>
#include <algorithm>
#include <chrono>
#include "MatrixLib/Matrix.hpp"
#include "NeuralNet.hpp"
#include "DataLoader.hpp"


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
            a = Matrix(vector<vector<double>>{{1, 2, 3}, {3, 4, 5}});
            (a.dot(a.T() + 6)).print();
            std::cout << "Power of matrix:\n";
            (a ^ 3.0).print();
            std::cout << "Loaded matrix:\n";
            vector<vector<double>> v {{2, 5}, {6, 1}, {3, 7}};
            Matrix b = Matrix(v);
            b.print();
            std::cout << "Argmax by axis 1:\n";
            (b.argmax(1)).print();
            std::cout << "Matrix mean:\n";
            (b.mean(0).mean(1)).print();
            std::cout << "Log matrix:\n";
            (b.log()).print();
            std::cout << "Reshaped matrix:\n";
            b = b.reshape(1, -1);
            b.print();
            std::cout << "Broadcasted matrix [1x1]:\n";
            a = (Matrix(1, 1) + 1).broadcast(std::make_tuple(2, 2));
            a.print();
            std::cout << "Elementwise addiction with broadcasting:\n";
            (a.reshape(-1, 1) + b).print();
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
            Matrix res = fc_layer.forward(x);
            res.print();
            x = Matrix(vector<vector<double>>{ { 1, -2, 3, 4 }, { -1, 2, 0.1, 1 } });
            fc_layer.backward(x).print();

            std::cout << std::fixed << "Optimizer check:\n";
            Matrix d_w( vector<vector<double>>{ { -0.980758, -0.980758, -0.980758, -0.980758 }, { 0.415141, 0.0565105, 0.415141, 0.415141 } } );
            nn::SGD<double> optim;
            optim.update(x, d_w, 0.9).print();

            std::cout << "Softmax check:\n";
            Matrix preds(vector<vector<double>>{ { 0.1, 0.2, 0.3, 0.4 }, { 0.4, 0.3, 0.2, 0.1 }});
            Matrix gt_ind(1, 1);
            gt_ind = Matrix(vector<vector<double>>{ { 3 }, { 1 }});
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
        
        // Real NN workcycle comes down here
        
        // Set fixed output precision
        std::cout << std::fixed;

        // Data loading
        DataLoader data_loader;

        DATASET_TYPE train_data = data_loader.load_dataset("data/train_32x32.dat", std::pair<int, int>(20000, 3072));
        DATASET_TYPE test_data = data_loader.load_dataset("data/test_32x32.dat", std::pair<int, int>(2000, 3072));

        data_loader.prepare_dataset(train_data.first, test_data.first);

        // Create and train model
        nn::Model model(3072, 10, 512, 1e-4);
        nn::Trainer trainer(model, train_data, new nn::SGD<double>(), 100, 400, 1e-1, 1.0);

        // Fit model and count the execution time
        auto t1 = std::chrono::high_resolution_clock::now();
        auto results = trainer.fit();
        auto t2 = std::chrono::high_resolution_clock::now();
        long double dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // Display the model runtime in a beautiful way
        std::cout << "Computed for " << std::defaultfloat;
        if (dur > 1000) {
            if (dur > 60000) {
                if (dur > 3600000) {
                    std::cout << dur / 3600000 << " hours\n";
                } else
                    std::cout << dur / 60000 << " minuts\n";
            } else
                std::cout << dur / 1000 << " seconds\n";
        } else
            std::cout << dur << " milliseconds\n";

        // Display the best results
        std::cout << "\nBest results:\nLoss: " << *std::min_element(results[0].begin(), results[0].end()) / 2 \
                << ", Train accuracy: " << *std::max_element(results[1].begin(), results[1].end()) \
                << ", Valid accuracy: " << *std::max_element(results[2].begin(), results[2].end()) << "\n";

        // Predict on test
        Matrix test_pred;
        vector<int> idx;
        for (int i = 0; i < (int)test_data.second.size(); i++)
            idx.push_back(i);
        auto test = data_loader.load_as_matrix(test_data.first, test_data.second, idx);
        test_pred = model.predict(test.first);
        // Compute the final score accuracy
        double test_accuracy = nn::Trainer::compute_accuracy(test_pred, test.second);
        std::cout << std::defaultfloat << "\nNeural net test accuracy: " << test_accuracy << "\n";
    }

    return 0;
}