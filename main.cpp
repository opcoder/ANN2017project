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

    const int layers_num = 3;
    vector<vector<double> > train_data = {
        {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1},
        {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
        {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1}
    };
    vector<vector<double> > train_label = {
        {1, -1, -1},
        {-1, 1, -1},
        {-1, -1, 1}
    }
    Net net = new Net(layers_num);

    vector<int> layers_dim = {16, 16, 16};
    vector<string> layers_filler = {gaussian_filler, gaussian_filler, gaussian_filler};
    vector<double> layers_lr = {1, 1, 1};
    vector<vector<double> > layers_filler_range = {{-1,1}, {-1,1}, {-1, 1}};
    net->InitNet(layers_dim, layers_filler, layers_lr, layers_filler_range);
    int max_iter = 1000;
    net->train(max_iter, train_data, train_label);

    return 0;
}
