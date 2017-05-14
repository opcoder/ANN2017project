#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <stdlib.h>
#include <math.h>

enum WeightFiller {
    Gaussian_filler,
    Uniform_filler
};
enum ActivateFunction {
    Sigmoid,
    Tanh,
    Identity
};

double gaussian_rand(double mu = 0, double sigma = 1.0);
std::vector<double> gaussian_filler(int N, double mu = 0, double sigma = 1.0);
std::vector<double> uniform_filler(int N, double min_value, double max_value);
double euclidean_distance(const std::vector<double> &A, const std::vector<double> &B);
inline double sigmoid(double x) {return 1. / (1. + exp(-x));}
#endif // UTILS_HPP_INCLUDED
