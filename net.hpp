#ifndef NET_HPP_INCLUDED
#define NET_HPP_INCLUDED

#include "layer.hpp"

class Net {
public:
    Net(int layer_num) {
        //layers_.resize(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layers_.push_back(new Layer());
        }
    }
    vector<double> Forward();
    vector<double> Backward();
    void InitNet(vector<int> layers_dim, vector<WeightFiller> fillers, vector<double>, 
        layers_lr, vector<vector<int> > filler_range);
    void train(int max_iter, vector<vector<double> > train_data,
        vector<vector<double> > train_label);
private:
    std::vector<Layer*> layers_;
};


#endif // NET_HPP_INCLUDED
