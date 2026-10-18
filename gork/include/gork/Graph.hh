#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/Tensor.hh"
#include "gork/ops.hh"
#include <algorithm>
#include <iterator>
#include <vector>

namespace gork {

using NodeId = size_t;

template <typename T>
class Graph {
  public:
    class Node {
      public:
        TensorOp op_;

        Tensor<T> tensor_;
        Tensor<T> gradient_;

        std::vector<NodeId> inputs_;

        Node(Tensor<T> tensor, TensorOp op);
        Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs);
    };

    Graph()
        : nodes_{} {}

    NodeId addInput(Tensor<T> &&tensor);
    NodeId addTensor(Tensor<T> &&tensor);

    NodeId matmul(NodeId lhs, NodeId rhs);
    NodeId broadcastAdd(NodeId lhs, NodeId rhs);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId tensor);

    Node &get(NodeId id);
    Tensor<T> &compute(NodeId toCompute);
    void backwards(NodeId toCompute);

    const std::vector<Node> &getNodes() const;

  private:
    std::vector<Node> nodes_;
};

template <typename T>
Graph<T>::Node::Node(Tensor<T> tensor, TensorOp op)
    : tensor_{std::move(tensor)},
      gradient_{std::vector<size_t>{0}},
      op_{op} {}

template <typename T>
Graph<T>::Node::Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs)
    : tensor_{std::move(tensor)},
      gradient_{std::vector<size_t>{0}},
      op_{op},
      inputs_{std::move(inputs)} {}

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
    Node &rightNode = nodes_[rhs];
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[0], rightNode.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::Matmul, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::broadcastAdd(NodeId lhs, NodeId rhs) {
    Node &rightNode = nodes_[rhs];
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[0], leftNode.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::BroadcastAdd, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
};

template <typename T>
NodeId Graph<T>::add(NodeId lhs, NodeId rhs) {
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{std::vector{leftNode.tensor_.shape_[0], leftNode.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::Add, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::relu(NodeId nodeId) {
    Node &node = nodes_[nodeId];

    Tensor<T> result{std::vector{node.tensor_.shape_[0], node.tensor_.shape_[1]}};
    nodes_.emplace_back(std::move(result), TensorOp::Relu, std::vector<NodeId>{nodeId});

    return nodes_.size() - 1;
};

template <typename T>
typename Graph<T>::Node &Graph<T>::get(NodeId id) {
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

        Node &node = get(nodeId);

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        Node &node = get(nodeId);
        switch (node.op_) {
        case TensorOp::Matmul: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::matmul(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Add: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::add(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Sub: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::sub(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::BroadcastAdd: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::broadcastAdd(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Relu: {
            Node &child0 = get(node.inputs_[0]);
            node.tensor_ = gork::broadcastRelu(child0.tensor_);
            break;
        }
        }
    }

    return get(toCompute).tensor_;
}

template <typename T>
void Graph<T>::backwards(NodeId toCompute) {
    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(toCompute);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        Node &node = get(nodeId);

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    Node &resultNode = get(toCompute);

    Tensor<T> resultGradient{std::vector<size_t>{}};
    resultGradient.at({}) = 1.0;
    resultNode.gradient_ = std::move(resultGradient);

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        Node &outNode = get(nodeId);

        switch (outNode.op_) {
        case TensorOp::Add:
        case TensorOp::BroadcastAdd: {
            Node &inode0 = get(outNode.inputs_[0]);
            Node &inode1 = get(outNode.inputs_[1]);

            inode0.gradient_ = outNode.gradient_;
            inode1.gradient_ = outNode.gradient_;

            break;
        }
        case TensorOp::Matmul: {
            Node &inNode0 = get(outNode.inputs_[0]);
            Node &inNode1 = get(outNode.inputs_[1]);

            inNode0.gradient_ = gork::matmul(outNode.gradient_, gork::transpose(inNode1.tensor_));
            inNode1.gradient_ = gork::matmul(gork::transpose(inNode0.tensor_), outNode.gradient_);

            break;
        }
        case TensorOp::Relu: {
            Node &inNode0 = get(outNode.inputs_[0]);

            inNode0.gradient_ =
                gork::elementwiseMul(gork::broadcastReluDeriv(inNode0.tensor_), outNode.gradient_);

            break;
        }
        }
    }
}

template <typename T>
const std::vector<typename Graph<T>::Node> &Graph<T>::getNodes() const {
    return nodes_;
}

} // namespace gork

#endif
