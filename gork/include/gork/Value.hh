#ifndef GORK_VALUE_H_
#define GORK_VALUE_H_

#include "gork/Tensor.hh"
#include <cstddef>

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
    union Val {
        Tensor<T> tensor_;
        size_t index_;
        T scalar_;

        Val() {};
        ~Val() {};
    } val_;

    Value();
    Value(T scalar);
    Value(Tensor<T> tensor);
    Value(size_t index);
    ~Value();

    Value(const Value &);
    Value(Value &&) noexcept;
    Value &operator=(const Value &);
    Value &operator=(Value &&) noexcept;

    Tensor<T> asTensor();

  private:
    void destroy() noexcept;
    void copyFrom(const Value &);
    void moveFrom(Value &&);
};

template <typename T>
Value<T>::Value()
    : ty_{ValueType::Undefined} {};

template <typename T>
Value<T>::Value(T scalar)
    : ty_{ValueType::Scalar} {
    val_.scalar_ = scalar;
};

template <typename T>
Value<T>::Value(Tensor<T> tensor)
    : ty_{ValueType::Tensor} {
    new (&val_.tensor_) Tensor<T>(std::move(tensor));
};

template <typename T>
Value<T>::Value(size_t index)
    : ty_{ValueType::Index} {
    val_.index_ = index;
};

template <typename T>
Value<T>::~Value() {
    destroy();
};

template <typename T>
void Value<T>::destroy() noexcept {
    switch (ty_) {
    case ValueType::Tensor:
        val_.tensor_.~Tensor();
        break;
    case ValueType::Undefined:
    case ValueType::Scalar:
    case ValueType::Index:
        break;
    }

    ty_ = ValueType::Undefined;
}

template <typename T>
void Value<T>::copyFrom(const Value &value) {
    switch (value.ty_) {
    case ValueType::Undefined:
        break;
    case ValueType::Scalar:
        val_.scalar_ = value.val_.scalar_;
        break;
    case ValueType::Index:
        val_.index_ = value.val_.index_;
        break;
    case ValueType::Tensor:
        new (&val_.tensor_) Tensor<T>(value.val_.tensor_);
        break;
    }

    ty_ = value.ty_;
}

template <typename T>
void Value<T>::moveFrom(Value &&value) {
    switch (value.ty_) {
    case ValueType::Undefined:
        break;
    case ValueType::Scalar:
        val_.scalar_ = value.val_.scalar_;
        break;
    case ValueType::Index:
        val_.index_ = value.val_.index_;
        break;
    case ValueType::Tensor:
        new (&val_.tensor_) Tensor<T>(std::move(value.val_.tensor_));
        break;
    }

    ty_ = value.ty_;
    value.ty_ = ValueType::Undefined;
}

template <typename T>
Value<T>::Value(const Value &value)
    : ty_{ValueType::Undefined} {
    copyFrom(value);
};

template <typename T>
Value<T>::Value(Value &&value) noexcept
    : ty_{ValueType::Undefined} {
    moveFrom(std::move(value));
};

template <typename T>
Value<T> &Value<T>::operator=(const Value &value) {
    if (this == &value)
        return *this;

    destroy();
    copyFrom(value);

    return *this;
}

template <typename T>
Value<T> &Value<T>::operator=(Value &&value) noexcept {
    if (this == &value)
        return *this;

    destroy();
    moveFrom(std::move(value));

    return *this;
}

template <typename T>
Tensor<T> Value<T>::asTensor() {
    switch (ty_) {
    case ValueType::Scalar: {
        Tensor<T> tensor{};
        tensor.at({}) = val_.scalar_;
        return tensor;
    }
    case ValueType::Tensor: {
        return val_.tensor_;
    }
    default:
        return Tensor<T>{};
    }
};

} // namespace gork

#endif
