#ifndef GORK_TENSOR_H_
#define GORK_TENSOR_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace gork {

template <typename T>
class Tensor {
  public:
    T *data_;
    size_t size_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    Tensor();
    Tensor(std::vector<size_t> tensor_shape);
    Tensor(std::initializer_list<size_t> tensor_shape);

    /// Create a new tensor with casted data
    template <typename R>
    Tensor(const Tensor<R> &);

    Tensor(const Tensor<T> &other);
    Tensor(Tensor<T> &&other) noexcept;
    Tensor<T> &operator=(const Tensor<T> &other);
    Tensor<T> &operator=(Tensor<T> &&other) noexcept;

    ~Tensor();

    bool isScalar() const;
    T at(std::initializer_list<size_t> index) const;
    T &at(std::initializer_list<size_t> index);

    void dump_mat();
};

template <typename T>
Tensor<T>::Tensor()
    : data_(nullptr) {}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape)
    : shape_{std::move(shape)} {
    size_ = 1;
    for (auto dim : shape_) {
        size_ *= dim;
    }

    data_ = new T[size_];

    size_t stride = 1;
    for (size_t i = 0; i < shape_.size(); i++) {
        strides_.push_back(stride);
        stride *= shape_[i];
    }
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<size_t> tensor_shape) {
    shape_.reserve(tensor_shape.size());
    for (size_t dim : tensor_shape) {
        shape_.push_back(dim);
    }

    size_ = 1;
    for (auto dim : shape_) {
        size_ *= dim;
    }

    data_ = new T[size_];

    size_t stride = 1;
    for (size_t i = 0; i < shape_.size(); i++) {
        strides_.push_back(stride);
        stride *= shape_[i];
    }
}

template <typename T>
template <typename R>
Tensor<T>::Tensor(const Tensor<R> &other) {
    shape_ = other.shape_;
    size_ = other.size_;
    strides_ = other.strides_;

    data_ = new T[size_];

    for (size_t i = 0; i < size_; i++) {
        data_[i] = static_cast<T>(other.data_[i]);
    }
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T> &rhs)
    : data_{nullptr},
      size_{rhs.size_},
      shape_{rhs.shape_},
      strides_{rhs.strides_} {
    if (rhs.data_ != nullptr) {
        data_ = new T[size_];
        std::copy(rhs.data_, rhs.data_ + size_, data_);
    }
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &&rhs) noexcept
    : data_{std::exchange(rhs.data_, nullptr)},
      size_{std::move(rhs.size_)},
      shape_{std::move(rhs.shape_)},
      strides_{std::move(rhs.strides_)} {}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &rhs) {
    if (this == &rhs)
        return *this;

    delete[] data_;

    size_ = rhs.size_;
    shape_ = rhs.shape_;
    strides_ = rhs.strides_;

    data_ = new T[size_];
    std::copy(rhs.data_, rhs.data_ + size_, data_);

    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&rhs) noexcept {
    if (this != &rhs) {
        data_ = std::move(rhs.data_);
        rhs.data_ = nullptr;
        shape_ = std::move(rhs.shape_);
        strides_ = std::move(rhs.strides_);
        size_ = rhs.size_;
    }

    return *this;
}

template <typename T>
Tensor<T>::~Tensor() {
    delete[] data_;
}

template <typename T>
bool Tensor<T>::isScalar() const {
    return size_ == 1;
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

// TODO: Make this able to print arbitrarily shaped tensors
template <typename T>
void Tensor<T>::dump_mat() {
    if (size_ == 1) {
        std::cout << "{ " << data_[0] << " }\n";
        return;
    }

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
