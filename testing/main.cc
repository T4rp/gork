#include "gork/tensor.hh"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::vector<size_t> shape{2, 2};
  gork::Tensor<float> tensor1{shape};
  gork::Tensor<float> tensor2{shape};

  tensor1.at({0, 0}) = 1;
  tensor1.at({1, 0}) = 2;
  tensor1.at({0, 1}) = 3;
  tensor1.at({1, 1}) = 4;

  tensor2.at({0, 0}) = 5;
  tensor2.at({1, 0}) = 6;
  tensor2.at({0, 1}) = 7;
  tensor2.at({1, 1}) = 8;

  tensor1.dump_mat();
  tensor2.dump_mat();

  gork::Tensor<float> tensor3 = gork::matmul(tensor1, tensor2);
  tensor3.dump_mat();

  std::cout << "hello world\n";
  return 0;
}
