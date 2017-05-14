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
    vector<double> layers_lr = {1, 1};
    vector<vector<double> > layers_filler_range = {{-1,1}, {-1,1}};
    vector<ActivateFunction> layers_activation = {Sigmoid, Identity};
    net->InitNet(layers_dim, layers_filler, layers_lr, layers_filler_range, layers_activation);
    int max_iter = 1000;
    int batch_size = 3;
    net->train(train_data, train_label, max_iter, batch_size);
    net->SaveModel("model.txt");

    return 0;
}
