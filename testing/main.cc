#include <iostream>
#include <vector>

#include "gork/tensor.hh"
#include "gork/ops.hh"

int main() {
    std::vector<size_t> inputShape{1, 2};

    std::vector<size_t> layer1Shape{2, 4};
    std::vector<size_t> layer1BiasShape{1, 4};

    std::vector<size_t> outputShape{4, 2};
    std::vector<size_t> outputBiasShape{1, 2};

    gork::Tensor<float> input{inputShape};
    gork::Tensor<float> layer1Weights{layer1Shape};
    gork::Tensor<float> layer1Bias{layer1BiasShape};

    gork::Tensor<float> outputWeights{outputShape};
    gork::Tensor<float> outputBias{outputBiasShape};

    gork::Tensor<float> a = gork::matmul(input, layer1Weights);
    gork::Tensor<float> b = gork::broadcastAdd(a, layer1Bias);
    gork::Tensor<float> c = gork::matmul(b, outputWeights);
    gork::Tensor<float> d = gork::broadcastAdd(c, outputBias);

    d.dump_mat();

    std::cout << "hello world\n";
    return 0;
}
