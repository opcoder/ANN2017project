#include "layer.hpp"

void Layer::Reshape(int bottom_dim, int top_dim) {
    // bottom_dim + 1 (offset)
    this->weights_.resize(top_dim, vector<double>(bottom_dim + 1, 0));
    this->bottom_data_.resize(bottom_dim + 1, 0);
    this->bottom_diff_.resize(bottom_dim + 1, 0);
    this->top_data_.resize(top_dim, 0);
    this->top_diff_.resize(top_dim, 0);
}
void Layer::Init(WeightFiller filler, double lr, vecotr<double> range, ActivateFunction activation) {

    this->activation_ = activation;
    this->lr_mult_ = lr;

    const int next_layer_dim = this->weights_.size();
    const int this_layer_dim = this->weights_[0].size() - 1;
    const int N = next_layer_dim * this_layer_dim;
    const double min_value = range[0], max_value = range[1];
    if (filler == gaussian_filler) {
        double mu = (range[0] + range[1]) / 2.0;
        double sigma = (range[1] - range[0]) / 6.0;
        vector<N> rand_filler = Gaussian_filler(N, mu, sigma);
        for (int i = 0, index = 0; i < next_layer_dim; ++i){
            for (int j = 0; j < this_layer_dim + 1; ++j, ++index) {
                double value = std::max(min_value, std::min(rand_filler[index], max_value));
                this->weights_[i][j] = value;
            }
        }
    } else {
        vector<N> rand_filler = Uniform_filler(N, min_value, max_value);
        for (int i = 0, index = 0; i < next_layer_dim; ++i){
            for (int j = 0; j < this_layer_dim + 1; ++j, ++index) {
                this->weights_[i][j] = rand_filler[index];
            }
        }
    }
}

vector<double> Layer::vector<double> Forward(const vector<double> &bottom_data) {
    this->bottom_data_ = bottom_data;

    assert(this->weights_.size() > 0);
    int output_dim = this->weights_.size();
    int input_dim = this->weights_[0].size() - 1;
    assert(bottom_data.size() == input_dim);
    for (int i = 0; i < output_dim; ++i) {
        this->top_data_[i] = this->weights_[i][0];
        for (int j = 0; j < input_dim; ++j) {
            this->top_data_[i] += bottom_data[j] * this->weights_[i][j+1]; 
        }
    }
    if (this->activation_ == Sigmoid) {
        for (int i = 0; i < top_data.size(); ++i) {
            this->top_data_[i] = sigmoid(this->top_data_[i]);
        }
    } if (this->activation_ == Tanh) {
        for (int i = 0; i < top_data.size(); ++i) {
            this->top_data_[i] = tanh(this->top_data_[i]);
        }
    }
    return this->top_data_;
}

void Layer::vector<double> Backward(const vector<double> &top_diff) {
    this->top_diff_ = top_diff;

    assert(this->weights_.size() > 0);
    int top_dim = this->weights_.size();
    int bottom_dim = this->weights_[0].size() - 1;
    assert(top_diff.size() == top_dim);


    if (this->activation_ == Sigmoid) {
        for (int i = 0; i < this->top_diff_.size(); ++i) {
            this->top_diff_[i] *= this->top_data_[i] * (1 - this->top_data_[i]);
        }
    } else if (this->activation_ == Tanh) {
        for (int i = 0; i < this->top_diff_.size(); ++i) {
            this->top_diff_[i] *= 1 - this->top_data_[i] * this->top_data_[i];
        }
    }

    this->bottom_diff_.resize(bottom_dim, 0);
    // Gradient with respect to bottom data
    for (int i = 0; i < top_dim; ++i) {
        for (int j = 0; j < bottom_dim; ++j) {
            this->bottom_diff_[j] += top_diff[i] * this->weights_[i][j+1];
        }
    }

    //accumlate Gradient with respect to weight
    for (int i = 0; i < top_dim; ++i) {
        double diff = top_diff[i];
        this->weights_diff_[i][0] += diff;
        for (int j = 0; j < bottom_dim; ++j) {
            this->weights_diff_[i][j + 1] += diff * this->bottom_data_[j];
        }
    }
    return bottom_diff;
}

//update wieghts and clear weights_diff_
void Layer::ApplyUpdate(int batch_size) {
    int top_dim = this->weights_.size();
    int bottom_dim = this->weights_[0].size() - 1;
    for (int i = 0; i < top_dim; ++i) {
        for (int j = 0; j < bottom_dim + 1; ++j) {
            this->weights_[i][j] -= lr/batch_size * this->weights_diff_[i][j];
            this->weights_diff_[i][j] = 0;
        }
    }
}