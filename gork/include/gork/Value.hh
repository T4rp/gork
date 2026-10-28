#ifndef GORK_VALUE_H_
#define GORK_VALUE_H_

#include "gork/Tensor.hh"
#include <algorithm>
#include <cstddef>
#include <variant>
#include <vector>

namespace gork {

enum class ValueType {
    Undefined,
    Scalar,
    Tensor,
    Index,
};

template <typename T>
class Value {
  public:
    ValueType ty_;
    std::variant<T, Tensor<T>, size_t> val_;

    Value();
    ~Value();

    Value(T scalar);
    Value(Tensor<T> tensor);
    Value(size_t index);

    Tensor<T> asTensor();
};

template <typename T>
Value<T>::Value()
    : ty_{ValueType::Undefined} {};

template <typename T>
Value<T>::Value(T scalar)
    : ty_{ValueType::Scalar},
      val_{scalar} {};

template <typename T>
Value<T>::Value(Tensor<T> tensor)
    : ty_{ValueType::Tensor},
      val_{std::move(tensor)} {};

template <typename T>
Value<T>::Value(size_t index)
    : ty_{ValueType::Index},
      val_{index} {};

template <typename T>
Value<T>::~Value() {
    ty_ = ValueType::Undefined;
};

template <typename T>
Tensor<T> Value<T>::asTensor() {
    switch (ty_) {
    case ValueType::Scalar: {
        Tensor<T> tensor{std::vector<size_t>{}};
        tensor.at({}) = std::get<T>(val_);
        return tensor;
    }
    case ValueType::Tensor: {
        return std::get<Tensor<T>>(val_);
    }
    default:
        return Tensor<T>{};
    }
};

} // namespace gork

#endif
