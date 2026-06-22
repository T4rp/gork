#include "gork/tensor.h"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::vector<size_t> shape{2, 2};
  gork::Tensor<float> tensor1{shape};
  gork::Tensor<float> tensor2{shape};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distr(0.0, 1.0);

  tensor1.at({1, 0}) = distr(gen);
  tensor1.at({1, 0}) = distr(gen);
  tensor1.at({0, 1}) = distr(gen);
  tensor1.at({1, 1}) = distr(gen);

  tensor2.at({1, 0}) = distr(gen);
  tensor2.at({1, 0}) = distr(gen);
  tensor2.at({0, 1}) = distr(gen);
  tensor2.at({1, 1}) = distr(gen);

  tensor1.dump_mat();

  // gork::Tensor<float> tensor3 = gork::matmul(tensor1, tensor2);

  // tensor3.dump_mat();

  std::cout << "hello world\n";
  return 0;
}
