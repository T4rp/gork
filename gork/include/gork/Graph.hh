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
        std::vector<NodeId> inputs_;

        /// Should only be used for constant nodes. treat as uninitalized otherwise
        Tensor<T> tensor_;

        Node(TensorOp op);
        Node(TensorOp op, std::vector<NodeId> inputs);
    };

    std::vector<NodeId> inputs_;
    std::vector<NodeId> outputs_;

    Graph()
        : nodes_{},
          inputs_{},
          outputs_{} {}

    NodeId input();
    Node &get(NodeId id);
    const std::vector<Node> &getNodes() const;

    NodeId constant(Tensor<T> tensor);
    NodeId output(NodeId nodeId);
    NodeId noOp(NodeId tensor);
    NodeId matmul(NodeId lhs, NodeId rhs);
    NodeId transpose(NodeId tensor);
    NodeId broadcastAdd(NodeId lhs, NodeId rhs);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId tensor);
    NodeId reluDeriv(NodeId tensor);

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
        result[nodeId] = constant(Tensor<T>{{1}});
        get(result[nodeId]).tensor_.at({}) = 0.0;

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    Node &resultNode = get(toCompute);

    auto intialGradient = constant(Tensor<T>{{1}});
    get(intialGradient).tensor_.at({}) = 1.0;

    result[toCompute] = intialGradient;

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

            result[inode0] = grad0;
            result[inode1] = grad1;

            break;
        }
        case TensorOp::Matmul: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = matmul(outNode, transpose(inode1));
            auto grad1 = matmul(transpose(inode0), outNode);

            result[inode0] = grad0;
            result[inode1] = grad1;

            break;
        }
        case TensorOp::Relu: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = mul(reluDeriv(inode0), outNode);

            result[inode0] = grad0;
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
