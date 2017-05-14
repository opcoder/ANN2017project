#include <cmath>
#include <cstdlib>

#incllude "utils.hpp"

double gaussian_rand(double mu, double sigma) {
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    return X * sigma + mu;
}

vector<double> gaussian_filler(int N, double mu = 0, double sigma = 1.0) {
    vector<double> filler;
    for (int i = 0; i < N; ++i) {
        filler.push_back(gaussian_rand(mu, sigma));
    }
    return filler;
}

vector<double> uniform_filler(int N, double min_value, double max_value) {
    vector<double> filler;
    for (int i = 0; i < N; ++i) {
        double x = (double)rand() / RAND_MAX;
        x = x / RAND_MAX * (max_value - min_value) + min_value;
        filler.push_back(x);
    }
    return filler;
}

double euclidean_distance(const vector<double> &A, const vector<double> &B) {
    double loss = 0;
    assert(A.size() == B.size());
    for (int i = 0; i < A.size(); ++i) {
        loss += (A[i] - B[i]) * (A[i] - B[i]) / 2.0;
    }
    return loss;
}
