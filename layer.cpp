#include "layer.hpp"

void Layer::Reshape(int bottom_dim, int top_dim) {
    // bottom_dim + 1 (offset)
    this->weights_.resize(top_dim, vector<double>(bottom_dim + 1, 0));
    this->bottom_data_.resize(bottom_dim + 1, 0);
    this->bottom_diff_.resize(bottom_dim + 1, 0);
}
void Layer::Init(WeightFiller filler, double lr, vecotr<double> range) {
    this->lr_mult_ = lr;
    const int next_layer_dim = this->weights_.size();
    const int this_layer_dim = this->weights_[0].size() - 1;
    const int N = next_layer_dim * this_layer_dim;
    const double min_value = range[0], max_value = range[1];
    if (filler == gaussian_filler) {
        double mu = (range[0] + range[1]) / 2.0;
        double sigma = (range[1] - range[0]) / 6.0;
        vector<N> rand_filler = gaussian_filler(N, mu, sigma);
        for (int i = 0, index = 0; i < next_layer_dim; ++i){
            for (int j = 0; j < this_layer_dim + 1; ++j, ++index) {
                double value = std::max(min_value, std::min(rand_filler[index], max_value));
                this->weights_[i][j] = value;
            }
        }
    } else {
        vector<N> rand_filler = uniform_filler(N, min_value, max_value);
        for (int i = 0, index = 0; i < next_layer_dim; ++i){
            for (int j = 0; j < this_layer_dim + 1; ++j, ++index) {
                this->weights_[i][j] = rand_filler[index];
            }
        }
    }
}

vector<double> Layer::vector<double> Forward(vector<double> &bottom_data) {
    assert(this->weights_.size() > 0);
    int output_dim = this->weights_.size();
    int input_dim = this->weights_[0].size() - 1;
    assert(bottom_data.size() == input_dim);
    this->bottom_data_ = bottom_data;
    vector<double> top_data(output_dim, 0);
    for (int i = 0; i < output_dim; ++i) {
        top_data[i] = this->weights_[i][0];
        for (int j = 0; j < input_dim; ++j) {
            top_data[i] += bottom_data[j] * this->weights_[i][j+1]; 
        }
    }
    return top_data;
}

void Layer::vector<double> Backward(vector<double> &top_diff) {
    assert(this->weights_.size() > 0);
    int top_dim = this->weights_.size();
    int bottom_dim = this->weights_[0].size() - 1;
    assert(top_diff.size() == top_dim);
    
    vector<double> bottom_diff(bottom_dim, 0);
    // Gradient with respect to bottom data
    for (int i = 0; i < top_dim; ++i) {
        for (int j = 0; j < bottom_dim; ++j) {
            bottom_diff[j] += top_diff[i] * this->weights_[i][j+1];
        }
    }

    // Gradient with respect to weight && update weight
    for (int i = 0; i < top_dim; ++i) {
        double diff = top_diff[i];
        this->weights_[i][0] -= lr * diff;
        for (int j = 0; j < bottom_dim; ++j) {
            this->weights_[i][j + 1] -= lr * diff * this->bottom_data_[j];
        }
    }
    return bottom_diff;
}