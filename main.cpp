#include <iostream>
#include <cassert>

#include "net.hpp"
using namespace std;

int main() {

//    int train_data[3][16] = {
//        {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1},
//        {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
//        {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1}
//    };
//    int train_label[3][3] = {
//        {1, -1, -1},
//        {-1, 1, -1},
//        {-1, -1, 1}
//    };

    vector<vector<double> > train_data = {
        {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1},
        {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
        {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1}
    };
    vector<vector<double> > train_label = {
        {1, -1, -1},
        {-1, 1, -1},
        {-1, -1, 1}
    };
    const int layers_num = 2;
    Net *net = new Net(layers_num);
    vector<int> layers_dim = {16, 16, 3}; // the last one is the output dimension
    vector<WeightFiller> layers_filler = {Gaussian_filler, Gaussian_filler};
    vector<double> layers_lr = {0.005, 0.005};
    vector<vector<double> > layers_filler_range = {{-1,1}, {-1,1}};
    vector<ActivateFunction> layers_activation = {Tanh, Identity};
//    vector<OptimizeAlgorithm> layers_opt_algorithm = {Adagrad, Adagrad};
    vector<OptimizeAlgorithm> layers_opt_algorithm = {Standard, Standard};
    net->InitNet(layers_dim, layers_filler, layers_lr, layers_filler_range, layers_activation, layers_opt_algorithm);

//    Net *net = new Net("model_9.txt");
    int max_iter = 1000000;
    int batch_size = 3;
    string model_name = "model_10.txt";
    string log_name = "log_10.txt";
    net->train(train_data, train_label, max_iter, -1, batch_size, log_name);
    net->SaveModel(model_name);
    delete net;
    return 0;
}
