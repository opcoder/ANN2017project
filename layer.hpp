//
// Created by wcbao on 2017/5/15.
//

#ifndef CLIONANN_LAYER_HPP
#define CLIONANN_LAYER_HPP

#include "utils.hpp"
using std::vector;

class Layer {
public:
    Layer(){
        this->activation_ = Tanh;
        this->apply_update_iter_ = 0;
        this->opt_algorithm_ = Adagrad;
    };
    Layer(ActivateFunction activation, vector<vector<double> > weights, OptimizeAlgorithm algorithm = Standard) {
        this->activation_ = activation;
        this->weights_ = weights;
        this->apply_update_iter_ = 0;
        this->opt_algorithm_ = Adagrad;
        int bottom_dim = this->weights_[0].size() - 1;
        int top_dim = this->weights_.size();
        Reshape(bottom_dim, top_dim);
    };

    void Init(WeightFiller filler, double lr, vector<double> range, ActivateFunction activation,
              OptimizeAlgorithm algorithm = Standard);
    vector<double> Forward(vector<double> bottom_data);
    vector<double> Backward(vector<double> top_diff);
    void ApplyUpdate(int batch_size);
    void Reshape(int bottom_dim, int top_dim);
    inline double get_weight(int i, int j) {
        assert(i < this->weights_.size());
        assert(j < this->weights_[0].size());
        return this->weights_[i][j];
    }
    inline ActivateFunction get_activation() {return this->activation_;}
    vector<int> get_weight_shape();
private:
    vector<vector<double> > weights_;
    vector<vector<double> > weights_diff_;
    vector<vector<double> > weights_diff_adam_m_;
    vector<vector<double> > weights_diff_adam_v_;
    vector<double> bottom_data_, bottom_diff_;
    vector<double> top_data_, top_diff_;
    double lr_mult_;
    ActivateFunction activation_;
    OptimizeAlgorithm opt_algorithm_;
    int apply_update_iter_;
    int bottom_dim_;
    int top_dim_;
};
#endif //CLIONANN_LAYER_HPP
