#ifndef GORK_TENSOR_H_
#define GORK_TENSOR_H_

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace gork {

template <typename T>
class Tensor {
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
Tensor<T>::Tensor(const std::vector<size_t> &tensor_shape) : shape(tensor_shape) {
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

template <typename T>
T &Tensor<T>::at(std::initializer_list<size_t> index) {
    assert(index.size() == shape.size());

    size_t offset = 0;
    size_t i = 0;

    for (auto idx : index) {
        offset += idx * strides[i];
        i++;
    }

    return data[offset];
}

template <typename T>
void Tensor<T>::dump_mat() {
    size_t m = shape[0];
    size_t n = shape[1];

    std::cout << "[\n";

    for (size_t i = 0; i < m; i++) {
        std::cout << "\t[";

        for (size_t j = 0; j < (n - 1); j++) {
            T val = at({i, j});
            std::cout << val << ", ";
        }

        T val = at({i, n - 1});

        std::cout << val << "],\n";
    }

    std::cout << "]\n";
}

} // namespace gork

#endif
