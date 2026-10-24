#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/Tensor.hh"
#include "gork/ops.hh"
#include <algorithm>
#include <iterator>
#include <stdexcept>
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
        std::vector<NodeId> inputs_;

        /// Should only be used for constant nodes. treat as uninitalized otherwise
        Tensor<T> tensor_;

        Node(TensorOp op);
        Node(TensorOp op, std::vector<NodeId> inputs);
    };

    std::vector<NodeId> inputs_;
    std::vector<NodeId> outputs_;

    Graph();

    Node &get(NodeId id);
    const std::vector<Node> &getNodes() const;

    NodeId noOp(NodeId tensor);
    NodeId input();
    NodeId output(NodeId nodeId);
    NodeId constant(Tensor<T> tensor);
    NodeId matmul(NodeId lhs, NodeId rhs);
    NodeId transpose(NodeId tensor);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId sub(NodeId lhs, NodeId rhs);
    NodeId mul(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId tensor);
    NodeId reluDeriv(NodeId tensor);
    NodeId negate(NodeId nodeId);
    NodeId vBroadcast(NodeId nodeId);
    NodeId vReduce(NodeId nodeId);

    std::unordered_map<NodeId, NodeId> mapGradientNodes(NodeId toCompute);

  private:
    std::vector<Node> nodes_;
};

template <typename T>
Graph<T>::Node::Node(TensorOp op)
    : tensor_{},
      op_{op} {}

template <typename T>
Graph<T>::Node::Node(TensorOp op, std::vector<NodeId> inputs)
    : tensor_{},
      op_{op},
      inputs_{std::move(inputs)} {}

template <typename T>
Graph<T>::Graph()
    : nodes_{},
      inputs_{},
      outputs_{} {}

template <typename T>
NodeId Graph<T>::input() {
    NodeId nextNodeId = nodes_.size();
    NodeId nextInputParamIndex = inputs_.size();

    nodes_.emplace_back(TensorOp::Input);

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextInputParamIndex;
    inputs_.push_back(nextNodeId);

    return nextNodeId;
}

template <typename T>
NodeId Graph<T>::constant(Tensor<T> tensor) {
    NodeId nextNodeId = nodes_.size();

    nodes_.emplace_back(TensorOp::Constant);

    Node &backNode = nodes_.back();
    backNode.tensor_ = std::move(tensor);

    return nextNodeId;
}

template <typename T>
NodeId Graph<T>::output(NodeId nodeId) {
    NodeId nextNodeId = nodes_.size();
    NodeId nextOutputParamIndex = outputs_.size();

    nodes_.emplace_back(TensorOp::Output, std::vector<NodeId>{nodeId});

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextOutputParamIndex;
    outputs_.push_back(nextNodeId);

    return nextNodeId;
}

template <typename T>
NodeId Graph<T>::noOp(NodeId tensor) {
    Node &node = nodes_[tensor];

    nodes_.emplace_back(TensorOp::NoOp, std::vector<NodeId>{tensor});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::matmul(NodeId lhs, NodeId rhs) {
    Node &rightNode = nodes_[rhs];
    Node &leftNode = nodes_[lhs];

    nodes_.emplace_back(TensorOp::Matmul, std::vector<NodeId>{lhs, rhs});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::transpose(NodeId tensor) {
    Node &node = nodes_[tensor];

    Tensor<T> result{};
    nodes_.emplace_back(TensorOp::Transpose, std::vector<NodeId>{tensor});

    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::add(NodeId lhs, NodeId rhs) {
    Node &leftNode = nodes_[lhs];
    nodes_.emplace_back(TensorOp::Add, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::sub(NodeId lhs, NodeId rhs) {
    Node &leftNode = nodes_[lhs];
    nodes_.emplace_back(TensorOp::Sub, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::mul(NodeId lhs, NodeId rhs) {
    Node &leftNode = nodes_[lhs];
    nodes_.emplace_back(TensorOp::Mul, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

template <typename T>
NodeId Graph<T>::relu(NodeId nodeId) {
    Node &node = nodes_[nodeId];

    nodes_.emplace_back(TensorOp::Relu, std::vector<NodeId>{nodeId});

    return nodes_.size() - 1;
};

template <typename T>
NodeId Graph<T>::reluDeriv(NodeId nodeId) {
    Node &node = nodes_[nodeId];

    nodes_.emplace_back(TensorOp::ReluDeriv, std::vector<NodeId>{nodeId});

    return nodes_.size() - 1;
};

template <typename T>
NodeId Graph<T>::negate(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::Negate, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

template <typename T>
typename Graph<T>::Node &Graph<T>::get(NodeId id) {
    return nodes_[id];
}

template <typename T>
std::unordered_map<NodeId, NodeId> Graph<T>::mapGradientNodes(NodeId toCompute) {
    std::unordered_map<NodeId, NodeId> result{};

    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(toCompute);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        Node &node = get(nodeId);
        result[nodeId] = constant(Tensor<T>{{}});
        get(result[nodeId]).tensor_.at({}) = 0.0;

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    Node &resultNode = get(toCompute);

    auto intialGradient = constant(Tensor<T>{{}});
    get(intialGradient).tensor_.at({}) = 1.0;

    result[toCompute] = intialGradient;

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        Node &outNode = get(nodeId);

        NodeId outNodeGradientId = result[nodeId];

        switch (outNode.op_) {
        case TensorOp::Input:
        case TensorOp::Constant:
            // root nodes
            break;
        case TensorOp::NoOp:
        case TensorOp::Output: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = noOp(outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::Matmul: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = matmul(outNodeGradientId, transpose(inode1));
            auto grad1 = matmul(transpose(inode0), outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);
            result[inode1] = add(result[inode1], grad1);

            break;
        }
        case TensorOp::Add: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = noOp(outNodeGradientId);
            auto grad1 = noOp(outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);
            result[inode1] = add(result[inode1], grad1);

            break;
        }
        case TensorOp::Sub: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = noOp(outNodeGradientId);
            auto grad1 = negate(outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);
            result[inode1] = add(result[inode1], grad1);

            break;
        }
        case TensorOp::Mul: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = mul(inode1, outNodeGradientId);
            auto grad1 = mul(inode0, outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);
            result[inode1] = add(result[inode1], grad1);

            break;
        }
        case TensorOp::Relu: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = mul(reluDeriv(inode0), outNodeGradientId);
            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::ReluDeriv:
            throw std::runtime_error("unimplemented");
        case TensorOp::VBroadcast:
        case TensorOp::VReduce:
            throw std::runtime_error("todo");
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
