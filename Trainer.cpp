#include "NeuralNet.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <memory>
#include "DataLoader.hpp"

nn::Trainer::Trainer (nn::Model &model,
                      DATASET_TYPE &dataset,
                      nn::Optim* optim,
                      int num_epochs,
                      int batch_size,
                      double learning_rate,
                      double learning_rate_decay) {
    this->model = model;
    this->dataset = dataset;
    this->optim = optim;
    this->num_epochs = num_epochs;
    this->batch_size = batch_size;
    this->learning_rate = learning_rate;
    this->learning_rate_decay = learning_rate_decay;
}


double nn::Trainer::compute_accuracy (Matrix &pred, Matrix &gt) {
    double accuracy, correct = 0;
    int size = std::get<0>(gt.shape());
    for (int i = 0; i < size; i++)
        if (pred(i, 0) == gt(i, 0))
            correct++;
    if (std::get<0>(pred.shape()) != 0)
        accuracy = correct / std::get<0>(pred.shape());
    else
        accuracy = 0;
    return accuracy;
}


vector<vector<int>> nn::Trainer::split_indices (vector<int> indices, int splits, bool shuffle) {
    // Shuffle batch indices if needed
    if (shuffle)
        std::random_shuffle(indices.begin(), indices.end());
    
    // Split shuffled indices for batches
    int size = (int)indices.size() / splits;

    vector<vector<int>> split_indices(splits);
    for (int i = 0; i < splits; i++) {
        auto first = std::next(indices.begin(), size * i);
        auto last = std::next(first, size);
        std::move(first, last, std::back_inserter(split_indices[i]));
    }

    return split_indices;
}


vector<vector<double>> nn::Trainer::fit () {
    // Setup optimizers for every param of the model
    vector<std::shared_ptr<Optim>> optimizers;
    for (auto it : model.get_params()){
        auto new_optim = optim->copy();
        optimizers.push_back(new_optim);
    }

    // Create validation folds
    vector<int> dataset_indices(dataset.first.size());
    for (int i = 0; i < (int)dataset.first.size(); i++){
        dataset_indices[i] = i;
    }
    auto val_indices = split_indices(dataset_indices, 10);

    // Sort validation indices to be able to easily exclue them out of train
    for (int i = 0; i < (int)val_indices.size(); i++) {
        std::sort(val_indices[i].begin(), val_indices[i].end());
    }

    vector<double> loss_history;
    vector<double> train_acc_history;
    vector<double> val_acc_history;

    // Set the timer
    auto t1 = std::chrono::high_resolution_clock::now();
    // Set the initial validation sample to be used
    int val_sample = 0; 

    for (int epoch = 0; epoch < num_epochs; epoch++){
        // Generate batch indices of dataset size avoiding validation indices
        vector<int> train_indices;
        int j = 0;
        for (int i = 0; i < (int)dataset.first.size(); i++){
            if (i != val_indices[val_sample][j])
                train_indices.push_back(i);
            else
                j++;
        }
        // Split batches by indices
        int splits;
        if (batch_size < (int)train_indices.size())
            splits = (int)train_indices.size() / batch_size;
        else
            splits = (int)train_indices.size();
        auto batch_indices = split_indices(train_indices, splits);

        // Iterate through all batches
        vector<double> batch_losses;
        for (auto batch : batch_indices) {

            // compute loss and gradients
            auto batch_values = DataLoader::load_as_matrix(dataset.first, dataset.second, batch);
            double loss = this->model.feed_forward(batch_values.first, batch_values.second);

            // optimize params
            auto params = model.get_params();
            for (int k = 0; k < (int)params.size(); k++){
                params[k]->value = optimizers[k]->update(params[k]->value, params[k]->grad, this->learning_rate);
            }

            batch_losses.push_back(loss);
        }

        // Perform learning rate decay
        this->learning_rate *= learning_rate_decay;

        // Compute average loss on all batches
        double avg_loss = std::accumulate(batch_losses.begin(), batch_losses.end(), 0.0) / (double)batch_losses.size();

        // predict and compute train accuracy
        auto train_values = DataLoader::load_as_matrix(dataset.first, dataset.second, train_indices);
        auto result = this->model.predict(train_values.first);
        double train_acc = compute_accuracy(result, train_values.second);

        // predict and compute validation accuracy
        auto val_values = DataLoader::load_as_matrix(dataset.first, dataset.second, val_indices[val_sample]);
        result = this->model.predict(val_values.first);
        double val_acc = compute_accuracy(result, val_values.second);

        // Compute the runtime of the neural network
        auto t2 = std::chrono::high_resolution_clock::now();
        double dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // Display the results of an epoch
        std::cout <<  "#" << epoch << " Train accuracy: " << train_acc \
            << ", val accuracy: " << val_acc << ", Time left: " << dur / 1000.0 << "\n";

        // Store the epoch results
        loss_history.push_back(avg_loss);
        train_acc_history.push_back(train_acc);
        val_acc_history.push_back(val_acc);

        // Update the validation sample
        if (val_sample < (int)val_indices.size() - 1)
            val_sample++;
        else
            val_sample = 0;
    }

    return vector<vector<double>>{loss_history, train_acc_history, val_acc_history};
}