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
    T *data_;
    size_t size_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    Tensor(const std::vector<size_t> &tensor_shape);
    ~Tensor();
    T at(std::initializer_list<size_t> index) const;
    T &at(std::initializer_list<size_t> index);

    void dump_mat();
};

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape) : shape_(shape) {
    size_ = 1;
    for (auto dim : shape) {
        size_ *= dim;
    }

    data_ = new T[size_];
    strides_.resize(shape.size());

    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape[i];
    }
}

template <typename T>
Tensor<T>::~Tensor() {
    delete[] data_;
}

template <typename T>
T Tensor<T>::at(std::initializer_list<size_t> index) const {
    assert(index.size() == shape_.size());

    size_t offset = 0;
    size_t i = 0;

    for (auto idx : index) {
        offset += idx * strides_[i];
        i++;
    }

    return data_[offset];
}

template <typename T>
T &Tensor<T>::at(std::initializer_list<size_t> index) {
    assert(index.size() == shape_.size());

    size_t offset = 0;
    size_t i = 0;

    for (auto idx : index) {
        offset += idx * strides_[i];
        i++;
    }

    return data_[offset];
}

template <typename T>
void Tensor<T>::dump_mat() {
    size_t m = shape_[0];
    size_t n = shape_[1];

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
