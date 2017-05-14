#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include "utils.hpp"

class Layer {
public:
    void Init(WeightFiller filler, double lr, vector<double> range);
    vector<double> Forward(vector<double> &bottom_data);
    vector<double> Backward(vector<double> &top_diff);
    void Reshape(int bottom_dim, int top_dim);
    void copy_to_bottom_data(vector<double> &from);
    void copy_to_top_diff(vector<double> &from);
private:
    vector<vector<double> > weights_;
    vector<vector<double> > bottom_data_, bottom_diff_;
    vector<vector<double> > top_data_, top_diff_;
    double lr_mult_ = 1.0;
};


#endif // LAYER_HPP_INCLUDED
