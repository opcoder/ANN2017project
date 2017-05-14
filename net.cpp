#include "net.hpp"

void Net::InitNet(vector<int> layers_dim, vector<WeightFiller> fillers, vector<double> layers_lr,
    vector<vector<double> > filler_range) {
    // layers_dim's last item is the output dimension
    assert(layers_dim.size() == this->layers_.size() + 1);
    for (int i = 0; i < this->layers_.size(); ++i) {
        this->layers_[i]->Reshape(layers_dim[i], layers_dim[i+1]);
        this->layers_[i]->Init(fillers[i], layers_lr[i], filler_range);
    }
}

void Net::train(int max_iter, vector<vector<double> > train_data, vector<vector<double> > train_label) {
    assert(train_data.size() == train_label.size());
    int batch = train_data.size();
    for (int i = 0; i < max_iter; ++i) {
        int index = i % batch;
        vector<double> data = train_data[index];
        vector<double> label = train_label[index];

        //forward
        vector<double> output = this->Forward(data);
        // calculate loss
        // double loss = euclidean_distance(output, label);
        vector<double> top_diff(output.size(), 0);
        for (int j = 0; j < output.size(); ++j) {
            top_diff[j] = output[j] - label[j];
        }
        
        this->Backward(top_diff);

    }
}


vector<double> Net::Forward(vector<double> bottom_data) {
    for (int i = 0; (int)i < this->layers_.size(); ++i) {
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

