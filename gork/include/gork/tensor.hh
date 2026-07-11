#ifndef GORK_TENSOR_H_
#define GORK_TENSOR_H_

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace gork {

template <typename T> class Tensor {
public:
  std::vector<T> data;
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  Tensor(const std::vector<size_t> &tensor_shape);
  T &at(std::initializer_list<size_t> index);
  Tensor<T> clone();

  void dump_mat();
};

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &tensor_shape)
    : shape(tensor_shape) {
  size_t data_size = 1;
  for (auto dim : shape) {
    data_size *= dim;
  }

  data.resize(data_size);
  strides.resize(shape.size());

  size_t stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

template <typename T> T &Tensor<T>::at(std::initializer_list<size_t> index) {
  assert(index.size() == shape.size());

  size_t offset = 0;
  size_t i = 0;

  for (auto idx : index) {
    offset += idx * strides[i];
    i++;
  }

  return data[offset];
}

template <typename T> void Tensor<T>::dump_mat() {
  size_t m = shape[0];
  size_t n = shape[1];

  std::cout << "[\n";

  for (size_t i = 0; i < m; i++) {
    std::cout << "\t[";

    for (size_t j = 0; j < n; j++) {
      T val = at({i, j});
      std::cout << val << ", ";
    }

    std::cout << "],\n";
  }

  std::cout << "]\n";
}

template <typename T> Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b) {
  size_t m = a.shape[0];
  size_t n = a.shape[1];
  size_t p = b.shape[1];

  assert(n == b.shape[0]);

  std::vector<size_t> size{m, p};
  Tensor<T> out{size};

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      T sum = 0;

      for (size_t k = 0; k < n; k++) {
        sum += a.at({i, k}) * b.at({k, j});
      }

      out.at({i, j}) = sum;
    }
  }

  return out;
}

template <typename T> void transposeMat(Tensor<T> &mat) {
  size_t shapeTemp{0};
  size_t strideTemp{0};

  shapeTemp = mat.shape[0];
  strideTemp = mat.strides[0];

  mat.shape[0] = mat.shape[1];
  mat.shape[1] = shapeTemp;

  mat.strides[0] = mat.strides[1];
  mat.strides[1] = strideTemp;
}

template <typename T> Tensor<T> broadcastAdd(Tensor<T> &mat, Tensor<T> &vec) {
  Tensor<T> tensor = mat;

  size_t m = tensor.shape[0];
  size_t n = tensor.shape[1];

  assert(n == vec.shape[1]);
  assert(vec.shape[0] == 1);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      tensor.at({i, j}) += vec.at({0, j});
    }
  }

  return tensor;
}

} // namespace gork

#endif
