#ifndef GORK_TENSOR_H_
#define GORK_TENSOR_H_

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace gork {

template <typename T> class Tensor {
public:
  Tensor(std::initializer_list<size_t> tensor_shape);

private:
  std::vector<T> data;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
};

template <typename T>
Tensor<T>::Tensor(std::initializer_list<size_t> tensor_shape)
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

} // namespace gork

#endif
