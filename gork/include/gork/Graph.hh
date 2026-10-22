#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/Tensor.hh"
#include "gork/ops.hh"
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace gork {

using NodeId = size_t;

template <typename T>
class Graph {
  public:
    class Node {
      public:
        TensorOp op_;
        size_t paramIndex_;

        Tensor<T> tensor_;
        Tensor<T> gradient_;

        std::vector<NodeId> inputs_;

        Node(Tensor<T> tensor, TensorOp op);
        Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs);
    };

    struct BackwardsGraphResult {
        /// graph for backwards pass
        Graph<T> graph;
        /// predicted value of forward pass
        NodeId prediction;
        /// expected value with inputs
        NodeId expected;
        /// input values
        std::vector<NodeId> inputs;
        /// maps forward nodes to backward nodes
        std::unordered_map<NodeId, NodeId> nodeMap;
    };

    std::vector<NodeId> inputs_;
    std::vector<NodeId> outputs_;

    Graph()
        : nodes_{},
          inputs_{},
          outputs_{} {}

    NodeId addInput(Tensor<T> tensor);
    Node &get(NodeId id);
    const std::vector<Node> &getNodes() const;

    NodeId output(NodeId nodeId);
    NodeId noOp(NodeId tensor);
    NodeId matmul(NodeId lhs, NodeId rhs);
    NodeId transpose(NodeId tensor);
    NodeId broadcastAdd(NodeId lhs, NodeId rhs);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId tensor);
    NodeId reluDeriv(NodeId tensor);

    Tensor<T> &compute(NodeId toCompute);

    void backwards(NodeId toCompute);
    BackwardsGraphResult buildBackwardsGraph(NodeId toCompute);

  private:
    std::vector<Node> nodes_;
};

template <typename T>
Graph<T>::Node::Node(Tensor<T> tensor, TensorOp op)
    : tensor_{std::move(tensor)},
      gradient_{},
      op_{op} {}

template <typename T>
Graph<T>::Node::Node(Tensor<T> tensor, TensorOp op, std::vector<NodeId> inputs)
    : tensor_{std::move(tensor)},
      gradient_{},
      op_{op},
      inputs_{std::move(inputs)} {}

template <typename T>
NodeId Graph<T>::addInput(Tensor<T> tensor) {
    NodeId nextNodeId = nodes_.size();
    NodeId nextInputParamIndex = inputs_.size();

    nodes_.emplace_back(std::move(tensor), TensorOp::Input);

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextInputParamIndex;
    inputs_.push_back(nextNodeId);

    return nextNodeId;
}

template <typename T>
NodeId Graph<T>::output(NodeId nodeId) {
    NodeId nextNodeId = nodes_.size();
    NodeId nextOutputParamIndex = outputs_.size();

    nodes_.emplace_back(Tensor<T>{}, TensorOp::Output, std::vector<NodeId>{nodeId});

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextOutputParamIndex;
    outputs_.push_back(nextNodeId);

    return nextNodeId;
}

template <typename T>
NodeId Graph<T>::noOp(NodeId tensor) {
    Node &node = nodes_[tensor];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::Matmul, std::vector<NodeId>{tensor});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::matmul(NodeId lhs, NodeId rhs) {
    Node &rightNode = nodes_[rhs];
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::Matmul, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::transpose(NodeId tensor) {
    Node &node = nodes_[tensor];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::Transpose, std::vector<NodeId>{tensor});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::broadcastAdd(NodeId lhs, NodeId rhs) {
    Node &rightNode = nodes_[rhs];
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::BroadcastAdd, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
};

template <typename T>
NodeId Graph<T>::add(NodeId lhs, NodeId rhs) {
    Node &leftNode = nodes_[lhs];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::Add, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::relu(NodeId nodeId) {
    Node &node = nodes_[nodeId];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::Relu, std::vector<NodeId>{nodeId});

    return nodes_.size() - 1;
};

template <typename T>
NodeId Graph<T>::reluDeriv(NodeId nodeId) {
    Node &node = nodes_[nodeId];

    Tensor<T> result{};
    nodes_.emplace_back(std::move(result), TensorOp::ReluDeriv, std::vector<NodeId>{nodeId});

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
            node.tensor_ = gork::ops::matmul(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Add: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::ops::add(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Sub: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::ops::sub(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::BroadcastAdd: {
            Node &child0 = get(node.inputs_[0]);
            Node &child1 = get(node.inputs_[1]);
            node.tensor_ = gork::ops::broadcastAdd(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Relu: {
            Node &child0 = get(node.inputs_[0]);
            node.tensor_ = gork::ops::relu(child0.tensor_);
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

            inNode0.gradient_ =
                gork::ops::matmul(outNode.gradient_, gork::ops::transpose(inNode1.tensor_));
            inNode1.gradient_ =
                gork::ops::matmul(gork::ops::transpose(inNode0.tensor_), outNode.gradient_);

            break;
        }
        case TensorOp::Relu: {
            Node &inNode0 = get(outNode.inputs_[0]);

            inNode0.gradient_ =
                gork::ops::mul(gork::ops::reluDeriv(inNode0.tensor_), outNode.gradient_);

            break;
        }
        }
    }
}

template <typename T>
typename Graph<T>::BackwardsGraphResult Graph<T>::buildBackwardsGraph(NodeId toCompute) {
    BackwardsGraphResult result{};

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

    auto intialGradient = result.graph.addTensor(Tensor<T>{});
    get(intialGradient).tensor_.at({}) = 1.0;

    result.nodeMap[toCompute] = intialGradient;

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        Node &outNode = get(nodeId);

        switch (outNode.op_) {
        case TensorOp::Add:
        case TensorOp::BroadcastAdd: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = noOp(nodeId);
            auto grad1 = noOp(nodeId);

            result.nodeMap[inode0] = grad0;
            result.nodeMap[inode1] = grad1;

            break;
        }
        case TensorOp::Matmul: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = matmul(outNode, transpose(inode1));
            auto grad1 = matmul(transpose(inode0), outNode);

            result.nodeMap[inode0] = grad0;
            result.nodeMap[inode1] = grad1;

            break;
        }
        case TensorOp::Relu: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = mul(reluDeriv(inode0), outNode);

            result.nodeMap[inode0] = grad0;
        }
        }
    }

    return result;
}

template <typename T>
const std::vector<typename Graph<T>::Node> &Graph<T>::getNodes() const {
    return nodes_;
}

} // namespace gork

#endif
