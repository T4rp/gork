#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/ops.hh"
#include "gork/tensor.hh"
#include <algorithm>
#include <optional>
#include <vector>

namespace gork {

using NodeId = size_t;

template <typename T>
class Node {
  public:
    TensorOp op_;

    Tensor<T> tensor_;
    std::optional<Tensor<T>> gradient_;

    std::vector<NodeId> inputs_;

    Node(Tensor<T> tensor, TensorOp op);
    Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs);
};

template <typename T>
Node<T>::Node(Tensor<T> tensor, TensorOp op) : tensor_{std::move(tensor)}, op_{op} {}

template <typename T>
Node<T>::Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs)
    : tensor_{std::move(tensor)}, op_{op}, inputs_{std::move(inputs)} {}

template <typename T>
class Graph {
  private:
    std::vector<Node<T>> nodes_;

  public:
    Graph() {}

    NodeId addInput(Tensor<T> &&tensor);
    NodeId matmul(NodeId rhs, NodeId lhs);

#ifdef GORK_TESTING
    const std::vector<Node<T>> &testNodes() const;
#endif
};

template <typename T>
NodeId Graph<T>::addInput(Tensor<T> &&tensor) {
    nodes_.emplace_back(std::move(tensor), TensorOp::Input);
    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::matmul(NodeId rhs, NodeId lhs) {
    Node<T> &rightNode = nodes_[rhs];
    Node<T> &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[1], rightNode.tensor_.shape_[0]}};
    nodes_.emplace_back(std::move(result), TensorOp::Matmul, std::vector<gork::NodeId>{rhs, lhs});

    return nodes_.size() - 1;
}

#ifdef GORK_TESTING
template <typename T>
const std::vector<Node<T>> &Graph<T>::testNodes() const {
    return nodes_;
}
#endif

} // namespace gork

#endif
