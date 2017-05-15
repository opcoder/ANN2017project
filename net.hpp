//
// Created by wcbao on 2017/5/15.
//

#ifndef CLIONANN_NET_HPP
#define CLIONANN_NET_HPP

#include "layer.hpp"
#include <fstream>
#include <vector>
#include <iostream>
#include <io.h>

using std::vector;
using std::string;
class Net {
public:
    Net(int layer_num) {
        //layers_.resize(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layers_.push_back(new Layer());
        }
    };
    Net(string model_name) {
        assert(access(model_name.c_str(), F_OK) == 0);
        std::ifstream reader(model_name);
        assert(reader.good());
        std::cout << "model_name:" << model_name << std::endl;
        int layer_num;
        reader >> layer_num;
        for (int layer_index = 0; layer_index < layer_num ; ++layer_index) {
            int activation, dim, weights_num;
            reader >> activation >> dim >> weights_num;
            int next_layer_dim = weights_num / dim;
            vector<vector<double> > weights(next_layer_dim, vector<double>(dim, 0));
            for (int i = 0; i < next_layer_dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    reader >> weights[i][j];
                }
            }
            this->layers_.push_back(new Layer(ActivateFunction(activation), weights));
        }
    };
    ~Net() {
        for (int i = 0; i < layers_.size(); ++i) {
            delete this->layers_[i];
        }
    }
    vector<double> Forward(vector<double> bottom_data);
    vector<double> Backward(vector<double> top_diff);
    void ApplyUpdate(int batch_size);
    void InitNet(vector<int> layers_dim, vector<WeightFiller> layers_filler,
                 vector<double> layers_lr, vector<vector<double> > layers_filler_range,
                 vector<ActivateFunction> layers_activation,
                 vector<OptimizeAlgorithm > layers_opt_algorithm);
    void train(vector<vector<double> > train_data,
               vector<vector<double> > train_label, int max_iter,
               double stop_loss, int batch_size, string log_name);
    void SaveModel(string model_name);
private:
    std::vector<Layer*> layers_;
};

#endif //CLIONANN_NET_HPP
