#ifndef GORK_OPS_H_
#define GORK_OPS_H_

#include "tensor.hh"

namespace gork {

enum class TensorOp {
    NoOp,
    Input,
    Matmul,
    Add,
    BroadcastAdd,
};

template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b) {
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

template <typename T>
Tensor<T> transposeMat(Tensor<T> &mat) {
    size_t m = mat.shape[1];
    size_t n = mat.shape[0];

    std::vector<size_t> outTensorShape{m, n};
    Tensor<T> outTensor{outTensor};

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            outTensor.at({j, i}) = mat.at({i, j});
        }
    }

    return outTensor;
}

template <typename T>
Tensor<T> broadcastAdd(Tensor<T> &mat, Tensor<T> &vec) {
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

template <typename T>
Tensor<T> broadcastReLU(Tensor<T> &tensor) {
    Tensor<T> newTensor = tensor;

    for (size_t i = 0; i < newTensor.data.size(); i++) {
        T out{0.0};

        if (newTensor.data[i] > 0) {
            out = newTensor.data[i];
        }

        newTensor.data[i] = out;
    }

    return newTensor;
}

} // namespace gork

#endif
