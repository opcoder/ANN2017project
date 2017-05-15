//
// Created by wcbao on 2017/5/15.
//
#include "net.hpp"
#include <numeric>
#include <iomanip>
#include <algorithm>
using std::vector;

void Net::InitNet(vector<int> layers_dim, vector<WeightFiller> layers_filler, vector<double> layers_lr,
                  vector<vector<double> > layers_filler_range,
                  vector<ActivateFunction> layers_activation,
                    vector<OptimizeAlgorithm > layers_opt_algorithm) {
    // the last item in layers_dim is the output dimension
    assert(layers_dim.size() == this->layers_.size() + 1);
    for (int i = 0; i < (int)this->layers_.size(); ++i) {
        this->layers_[i]->Reshape(layers_dim[i], layers_dim[i+1]);
        this->layers_[i]->Init(layers_filler[i], layers_lr[i], layers_filler_range[i],
                               layers_activation[i], layers_opt_algorithm[i]);
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
void Net::train(vector<vector<double> > train_data, vector<vector<double> > train_label, int max_iter,
                double stop_loss, int batch_size, string log_name) {
    assert(train_data.size() == train_label.size());
    if (batch_size <= 0) {
        batch_size = train_data.size();
    }
    const int ave_loss_iter = 100;
    vector<double> losses;
    double smoothed_loss = 0;
    std::ofstream log(log_name);
    double batch_loss = 0;
    for (int i = 0; i < max_iter; ++i) {
        int index = i % batch_size;
        vector<double> data = train_data[index];
        vector<double> label = train_label[index];
        //step 1: forward
        vector<double> output = this->Forward(data);
        double loss = euclidean_distance(output, label);

        if (index == 0) {
            batch_loss = loss;
        } else {
            batch_loss += loss;
        }
        if (losses.size() < ave_loss_iter) {
//            smoothed_loss = (smoothed_loss * i + loss) / (i+1);
            losses.push_back(loss);
        } else {
            int idx = i % ave_loss_iter;
//            smoothed_loss = (smoothed_loss * ave_loss_iter - losses[idx] + loss) / ave_loss_iter;
            losses[idx] = loss;
        }
        double sum_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
        smoothed_loss =  sum_loss / losses.size();
        //step 2: calculate diff
        vector<double> top_diff(output.size(), 0);
        for (int j = 0; j < (int)output.size(); ++j) {
            top_diff[j] = output[j] - label[j];
        }
        //step 3: backward
        this->Backward(top_diff);
        //step 4: update
        if (index == batch_size - 1) {
            if (batch_loss/batch_size < stop_loss) {
                std::cout << "epoch " << i/batch_size << " loss:" << batch_loss/batch_size
                          << ", smoothed_loss :" << smoothed_loss << std::endl;
                log << "epoch " << i/batch_size << " loss:" << batch_loss/batch_size
                    << ", smoothed_loss :" << smoothed_loss << std::endl;
                log.close();
                return ;
            }
            std::cout << "epoch " << i/batch_size + 1 << " loss:" << batch_loss/batch_size
                    << ", smoothed_loss :" << smoothed_loss << std::endl;
            log << "epoch " << i/batch_size + 1 << " loss:" << batch_loss/batch_size
                    << ", smoothed_loss :" << smoothed_loss << std::endl;
            if (i >= 9000) {
                double x = fast_power(2, 10);
                x = x + 1;
            }
            this->ApplyUpdate(batch_size);
        }
    }
    log.close();
}

void Net::SaveModel(string model_name) {
    std::ofstream writer(model_name);
    writer << std::fixed;
    writer << this->layers_.size() << std::endl;
    for (int layer_index = 0; layer_index < (int)this->layers_.size(); ++layer_index) {
        Layer *layer = this->layers_[layer_index];
        vector<int> shape = layer->get_weight_shape();
        int dim = shape[1];
        int next_layer_dim = shape[0];
        writer << layer->get_activation() << " " << dim + 1 << " " << (dim + 1) * next_layer_dim << " ";
        for (int i = 0; i < next_layer_dim; ++i) {
            for (int j = 0; j < dim + 1; ++j) {
                writer << layer->get_weight(i, j) << " ";
            }
        }
        writer << std::endl;
    }
    writer.close();
    std::cout << "save model as " << model_name << std::endl;
}

