#include "net.hpp"
#include <iostream>
using std::vector;

void Net::InitNet(vector<int> layers_dim, vector<WeightFiller> layers_filler, vector<double> layers_lr,
    vector<vector<double> > layers_filler_range, vector<ActivateFunction> layers_activation) {
    // the last item in layers_dim is the output dimension
    assert(layers_dim.size() == this->layers_.size() + 1);
    for (int i = 0; i < (int)this->layers_.size(); ++i) {
        this->layers_[i]->Reshape(layers_dim[i], layers_dim[i+1]);
        this->layers_[i]->Init(layers_filler[i], layers_lr[i], layers_filler_range[i], layers_activation[i]);
    }
}

vector<double> Net::Forward(vector<double> bottom_data) {
    for (int i = 0; i < (int)this->layers_.size(); ++i) {
        bottom_data = this->layers_[i]->Forward(bottom_data);
    }
    //network's output actually
    return bottom_data;
}

vector<double> Net::Backward(vector<double> top_diff) {
    for (int i = this->layers_.size() - 1; i >= 0; --i) {
        top_diff = this->layers_[i]->Backward(top_diff);
    }
    return top_diff;
}

void Net::ApplyUpdate(int batch_size) {
    for (int i = 0; i < (int)this->layers_.size(); ++i) {
        this->layers_[i]->ApplyUpdate(batch_size);
    }
}
void Net::train(vector<vector<double> > train_data, vector<vector<double> > train_label, int max_iter, int batch_size) {
    assert(train_data.size() == train_label.size());
    if (batch_size <= 0) {
        batch_size = train_data.size();
    }
    for (int i = 0; i < max_iter; ++i) {
        int index = i % batch_size;
        vector<double> data = train_data[index];
        vector<double> label = train_label[index];
        //step 1: forward
        vector<double> output = this->Forward(data);
        // double loss = euclidean_distance(output, label);
        //step 2: calculate diff
        vector<double> top_diff(output.size(), 0);
        for (int j = 0; j < (int)output.size(); ++j) {
            top_diff[j] = output[j] - label[j];
        }
        //step 3: backward
        this->Backward(top_diff);
        //step 4: update
        if (index == batch_size - 1) {
            this->ApplyUpdate(batch_size);
        }
    }
}

void Net::SaveModel(string model_name) {
    std::ofstream writer(model_name);
    writer << this->layers_.size() << std::endl;
    for (int layer_index = 0; layer_index < (int)this->layers_.size(); ++layer_index) {
        Layer *layer = this->layers_[layer_index];
        vector<int> shape = layer->get_weight_shape();
//        int dim = layer->weights_[0].size() - 1;
//        int next_layer_dim = layer->weights_.size();
        int dim = shape[1];
        int next_layer_dim = shape[0];
        writer << layer->get_activation() << " " << dim << " " << dim * next_layer_dim << " ";
        for (int i = 0; i < next_layer_dim; ++i) {
            for (int j = 0; j < dim + 1; ++j) {
                writer << layer->get_weight(i, j) << " ";
            }
        }
        writer << std::endl;
    }
    std::cout << "save model as " << model_name << std::endl;
}
