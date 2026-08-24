#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/ops.hh"
#include "gork/tensor.hh"
#include <algorithm>
#include <iterator>
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
    NodeId addTensor(Tensor<T> &&tensor);
    NodeId matmul(NodeId rhs, NodeId lhs);
    NodeId add(NodeId rhs, NodeId lhs);
    Node<T> &get(NodeId id);
    Tensor<T> &compute(NodeId toCompute);

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
NodeId Graph<T>::addTensor(Tensor<T> &&tensor) {
    nodes_.emplace_back(std::move(tensor), TensorOp::NoOp);
    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::matmul(NodeId lhs, NodeId rhs) {
    Node<T> &rightNode = nodes_[rhs];
    Node<T> &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[0], rightNode.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::Matmul, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::add(NodeId lhs, NodeId rhs) {
    Node<T> &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[0], leftNode.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::Add, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
Node<T> &Graph<T>::get(NodeId id) {
    return nodes_[id];
}

template <typename T>
Tensor<T> &Graph<T>::compute(NodeId toCompute) {
    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(toCompute);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        Node<T> &node = get(nodeId);

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        Node<T> node = get(nodeId);
        if (node.op_ == TensorOp::Matmul) {
            Node<T> child0 = get(node.inputs_[0]);
            Node<T> child1 = get(node.inputs_[1]);
            node.tensor_ = gork::matmul(child0.tensor_, child1.tensor_);
        }
    }

    return get(toCompute).tensor_;
}

#ifdef GORK_TESTING
template <typename T>
const std::vector<Node<T>> &Graph<T>::testNodes() const {
    return nodes_;
}
#endif

} // namespace gork

#endif
