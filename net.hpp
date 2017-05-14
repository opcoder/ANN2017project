#ifndef NET_HPP_INCLUDED
#define NET_HPP_INCLUDED

#include "layer.hpp"
#include <fstream>

class Net {
public:
    Net(int layer_num) {
        //layers_.resize(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layers_.push_back(new Layer());
        }
    }
    Net(string model_name) {
        std::ifstream reader(model_name);
        int layer_num;
        reader >> layer_num;
        for (int layer_index = 0; layer_index < layer_num ; ++layer_index) {
            int activation, dim, weights_num;
            reader >> activation >> dim >> weights_num;
            int next_layer_dim = weights_num / dim;
            vector<vector<double> > weights;
            for (int i = 0; i < next_layer_dim; ++i) {
                for (int j = 0; j < dim + 1; ++j) {
                    reader >> weights[i][j];
                }
            }
            this->layers_.push_back(new Layer(ActivateFunction(activation), weights));
        }
    }
    vector<double> Forward();
    vector<double> Backward();
    void ApplyUpdate(int batch_size);
    void InitNet(vector<int> layers_dim, vector<WeightFiller> layers_filler, vector<double>, 
        layers_lr, vector<vector<int> > layers_filler_range, ActivateFunction layers_activation);
    void train(vector<vector<double> > train_data,
        vector<vector<double> > train_label, int max_iter, int batch_size = -1);
    void save_model(string model_name);
private:
    std::vector<Layer*> layers_;
};


#endif // NET_HPP_INCLUDED
