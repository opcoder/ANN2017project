#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include "utils.hpp"
using std::vector;

class Layer {
public:
    Layer(){};
    Layer(ActivateFunction activation, vector<vector<double> > weights) {
        this->activation_ = activation;
        this->weights_ = weights;
    };

    void Init(WeightFiller filler, double lr, vector<double> range, ActivateFunction activation);
    vector<double> Forward(vector<double> bottom_data);
    vector<double> Backward(vector<double> top_diff);
    void ApplyUpdate(int batch_size);
    void Reshape(int bottom_dim, int top_dim);
    void copy_to_bottom_data(vector<vector<double> > &from);
    void copy_to_top_diff(vector<vector<double> > &from);
    inline double get_weight(int i, int j) {return this->weights_[i][j];}
    inline ActivateFunction get_activation() {return this->activation_;}
    vector<int> get_weight_shape();
private:
    vector<vector<double> > weights_;
    vector<vector<double> > weights_diff_;
    vector<double> bottom_data_, bottom_diff_;
    vector<double> top_data_, top_diff_;
    double lr_mult_;
    ActivateFunction activation_;
    int bottom_dim_;
    int top_dim_;
};

#endif // LAYER_HPP_INCLUDED
