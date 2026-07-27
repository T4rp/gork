#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/ops.hh"
#include "gork/tensor.hh"
#include <vector>

namespace gork {

using NodeId = size_t;

template <typename T>
class Node {
  public:
    TensorOp op_;

    Tensor<T> tensor_;
    Tensor<T> gradient_;

    std::vector<NodeId> inputs_;

    Node(Tensor<T> tensor, TensorOp op) : op_{op}, tensor_{std::move(tensor)} {}
};

template <typename T>
class Graph {
  public:
    std::vector<Node<T>> nodes_;

    NodeId &addInput(Tensor<T> &&tensor) {
        nodes_.emplace_back({std::move(tensor), TensorOp::Input});
        return nodes_.size() - 1;
    }
};

} // namespace gork

#endif
