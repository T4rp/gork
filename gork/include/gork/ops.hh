#ifndef GORK_OPS_H_
#define GORK_OPS_H_

#include "Tensor.hh"
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gork {

enum class TensorOp {
    NoOp,
    Input,
    Matmul,
    Transpose,
    Add,
    Sub,
    Mul,
    ElementwiseSquare,
    BroadcastAdd,
    Relu,
    ReluDeriv,

    Reduce,
};

// TODO: support multiple dimensions
template <typename T>
Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
    size_t m = a.shape_[0];
    size_t n = a.shape_[1];
    size_t p = b.shape_[1];

    assert(n == b.shape_[0]);

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

template <typename T>
Tensor<T> transpose(const Tensor<T> &mat) {
    size_t m = mat.shape_[1];
    size_t n = mat.shape_[0];

    std::vector<size_t> outTensorShape{m, n};
    Tensor<T> outTensor{outTensor.shape_};

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            outTensor.at({j, i}) = mat.at({i, j});
        }
    }

    return outTensor;
}

template <typename T>
Tensor<T> add(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    if (lhs.isScalar() && rhs.isScalar()) {
        Tensor<T> tensor{};
        tensor.data_[0] = lhs.data_[0] + rhs.data_[0];
        return tensor;
    } else if (rhs.isScalar()) {
        Tensor<T> tensor{lhs.shape_};
        T scalar = rhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] + scalar;
        }
        return tensor;
    } else if (lhs.isScalar()) {
        Tensor<T> tensor{rhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < rhs.size_; i++) {
            tensor.data_[i] = scalar + rhs.data_[i];
        }
        return tensor;
    } else {
        if (lhs.shape_.size() != rhs.shape_.size())
            throw std::runtime_error("Incompatible tensor shape");

        for (size_t i = 0; i < lhs.shape_.size(); i++) {
            if (lhs.shape_[i] != rhs.shape_[i])
                throw std::runtime_error("Incompatible tensor shape");
        }

        Tensor<T> tensor{lhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] + rhs.data_[i];
        }
        return tensor;
    };
}

template <typename T>
Tensor<T> sub(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    if (lhs.isScalar() && rhs.isScalar()) {
        Tensor<T> tensor{};
        tensor.data_[0] = lhs.data_[0] - rhs.data_[0];
        return tensor;
    } else if (rhs.isScalar()) {
        Tensor<T> tensor{lhs.shape_};
        T scalar = rhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] - scalar;
        }
        return tensor;
    } else if (lhs.isScalar()) {
        Tensor<T> tensor{rhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < rhs.size_; i++) {
            tensor.data_[i] = scalar - rhs.data_[i];
        }
        return tensor;
    } else {
        if (lhs.shape_.size() != rhs.shape_.size())
            throw std::runtime_error("Incompatible tensor shape");

        for (size_t i = 0; i < lhs.shape_.size(); i++) {
            if (lhs.shape_[i] != rhs.shape_[i])
                throw std::runtime_error("Incompatible tensor shape");
        }

        Tensor<T> tensor{lhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] - rhs.data_[i];
        }
        return tensor;
    }
}

template <typename T>
Tensor<T> mul(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    if (lhs.isScalar() && rhs.isScalar()) {
        Tensor<T> tensor{};
        tensor.data_[0] = lhs.data_[0] * rhs.data_[0];
        return tensor;
    } else if (rhs.isScalar()) {
        Tensor<T> tensor{lhs.shape_};
        T scalar = rhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] * scalar;
        }
        return tensor;
    } else if (lhs.isScalar()) {
        Tensor<T> tensor{rhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < rhs.size_; i++) {
            tensor.data_[i] = scalar * rhs.data_[i];
        }
        return tensor;
    } else {
        if (lhs.shape_.size() != rhs.shape_.size())
            throw std::runtime_error("Incompatible tensor shape");

        for (size_t i = 0; i < lhs.shape_.size(); i++) {
            if (lhs.shape_[i] != rhs.shape_[i])
                throw std::runtime_error("Incompatible tensor shape");
        }

        Tensor<T> tensor{lhs.shape_};
        T scalar = lhs.data_[0];
        for (size_t i = 0; i < lhs.size_; i++) {
            tensor.data_[i] = lhs.data_[i] * rhs.data_[i];
        }
        return tensor;
    }
}

template <typename T>
Tensor<T> elementwiseSquare(const Tensor<T> &input) {
    if (input.isScalar()) {
        Tensor<T> tensor{};
        tensor.data_[0] = input.data_[0] * input.data_[0];
        return tensor;
    } else {
        Tensor<T> tensor{input.shape_};
        for (size_t i = 0; i < input.size_; i++) {
            tensor.data_[i] = input.data_[i] * *input.data_[i];
        }
        return tensor;
    }
}

template <typename T>
Tensor<T> broadcastAdd(const Tensor<T> &mat, const Tensor<T> &vec) {
    Tensor<T> tensor = mat;

    size_t m = tensor.shape_[0];
    size_t n = tensor.shape_[1];

    assert(m == vec.shape_[0]);
    assert(vec.shape_[1] == 1);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            tensor.at({i, j}) += vec.at({i, 0});
        }
    }

    return tensor;
}

template <typename T>
Tensor<T> relu(const Tensor<T> &tensor) {
    Tensor<T> newTensor = tensor;

    for (size_t i = 0; i < newTensor.size_; i++) {
        T out{0.0};

        if (newTensor.data_[i] > 0) {
            out = newTensor.data_[i];
        }

        newTensor.data_[i] = out;
    }

    return newTensor;
}

template <typename T>
Tensor<T> reluDeriv(const Tensor<T> &tensor) {
    Tensor<T> newTensor = tensor;

    for (size_t i = 0; i < newTensor.size_; i++) {
        T out{0.0};

        if (newTensor.data_[i] > 0) {
            out = 1.0;
        }

        newTensor.data_[i] = out;
    }

    return newTensor;
}

} // namespace gork

#endif
