//
// Created by wcbao on 2017/5/15.
//

#ifndef CLIONANN_UTILS_HPP
#define CLIONANN_UTILS_HPP

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cassert>
#include <cstring>

using std::vector;
enum WeightFiller {
    Gaussian_filler,
    Uniform_filler
};
enum ActivateFunction {
    Sigmoid,
    Tanh,
    Identity
};
enum OptimizeAlgorithm {
    Standard,
    Adagrad
};

double gaussian_rand(double mu = 0, double sigma = 1.0);
vector<double> gaussian_filler(int N, double mu = 0, double sigma = 1.0);
vector<double> uniform_filler(int N, double min_value, double max_value);
double euclidean_distance(const std::vector<double> &A, const std::vector<double> &B);
inline double sigmoid(double x) {return 1. / (1. + exp(-x));}
double fast_power(double x, int p);
#endif //CLIONANN_UTILS_HPP
