#include "gork/Graph.hh"
#include "gork/Value.hh"
#include <cassert>

namespace gork {

Graph::Node::Node(TensorOp op)
    : op_{op},
      value_{} {}

Graph::Node::Node(TensorOp op, std::vector<NodeId> inputs)
    : op_{op},
      inputs_{std::move(inputs)},
      value_{} {}

Graph::Graph()
    : inputs_{},
      outputs_{},
      nodes_{} {}

NodeId Graph::input() {
    NodeId nextNodeId = nodes_.size();
    NodeId nextInputParamIndex = inputs_.size();

    nodes_.emplace_back(TensorOp::Input);

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextInputParamIndex;
    inputs_.push_back(nextNodeId);

    return nextNodeId;
}

NodeId Graph::constant(Value<float> value) {
    NodeId nextNodeId = nodes_.size();

    nodes_.emplace_back(TensorOp::Constant);

    Node &backNode = nodes_.back();
    backNode.value_ = std::move(value);

    return nextNodeId;
}

NodeId Graph::output(NodeId nodeId) {
    NodeId nextNodeId = nodes_.size();
    NodeId nextOutputParamIndex = outputs_.size();

    nodes_.emplace_back(TensorOp::Output, std::vector<NodeId>{nodeId});

    auto &backNode = nodes_.back();
    backNode.paramIndex_ = nextOutputParamIndex;
    outputs_.push_back(nextNodeId);

    return nextNodeId;
}

NodeId Graph::noOp(NodeId tensor) {
    nodes_.emplace_back(TensorOp::NoOp, std::vector<NodeId>{tensor});
    return nodes_.size() - 1;
}

NodeId Graph::matmul(NodeId lhs, NodeId rhs) {
    nodes_.emplace_back(TensorOp::Matmul, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

NodeId Graph::transpose(NodeId tensor) {
    nodes_.emplace_back(TensorOp::Transpose, std::vector<NodeId>{tensor});
    return nodes_.size() - 1;
}

NodeId Graph::add(NodeId lhs, NodeId rhs) {
    nodes_.emplace_back(TensorOp::Add, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

NodeId Graph::sub(NodeId lhs, NodeId rhs) {
    nodes_.emplace_back(TensorOp::Sub, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

NodeId Graph::mul(NodeId lhs, NodeId rhs) {
    nodes_.emplace_back(TensorOp::Mul, std::vector<NodeId>{lhs, rhs});
    return nodes_.size() - 1;
}

NodeId Graph::elemSquare(NodeId nodeId) {
nodes_.emplace_back(TensorOp::ElementwiseSquare, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
}

NodeId Graph::relu(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::Relu, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

NodeId Graph::reluDeriv(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::ReluDeriv, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

NodeId Graph::negate(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::Negate, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

NodeId Graph::hBroadcast(NodeId nodeId, size_t count) {
    auto countNodeId = constant(Value<float>{count});
    nodes_.emplace_back(TensorOp::HBroadcast, std::vector<NodeId>{nodeId, countNodeId});
    return nodes_.size() - 1;
};

NodeId Graph::hBroadcast2(NodeId nodeId, NodeId countNode) {
    nodes_.emplace_back(TensorOp::HBroadcast, std::vector<NodeId>{nodeId, countNode});
    return nodes_.size() - 1;
};

NodeId Graph::hReduce(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::HReduce, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

NodeId Graph::vBroadcast(NodeId nodeId, size_t count) {
    auto countNodeId = constant(Value<float>{count});
    nodes_.emplace_back(TensorOp::VBroadcast, std::vector<NodeId>{nodeId, countNodeId});
    return nodes_.size() - 1;
};

NodeId Graph::vBroadcast2(NodeId nodeId, NodeId countNode) {
    nodes_.emplace_back(TensorOp::VBroadcast, std::vector<NodeId>{nodeId, countNode});
    return nodes_.size() - 1;
};

NodeId Graph::vReduce(NodeId nodeId) {
    nodes_.emplace_back(TensorOp::VReduce, std::vector<NodeId>{nodeId});
    return nodes_.size() - 1;
};

NodeId Graph::getDim(NodeId nodeId, size_t dimIndex) {
    auto countNodeId = constant(Value<float>{dimIndex});

    nodes_.emplace_back(TensorOp::GetDim, std::vector<NodeId>{nodeId, countNodeId});
    return nodes_.size() - 1;
}

typename Graph::Node &Graph::get(NodeId id) {
    return nodes_[id];
}

std::unordered_map<NodeId, NodeId> Graph::mapGradientNodes(NodeId toCompute) {
    std::unordered_map<NodeId, NodeId> result{};

    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(toCompute);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        Node &node = get(nodeId);
        result[nodeId] = constant(Value{0.0f});

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    auto intialGradient = constant(Value{1.0f});
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

            // the add(mul(in, 0.0), grad) step is used to shape the incoming gradient
            auto grad0 = add(mul(inode0, constant(Value<float>{0.0f})), outNodeGradientId);
            auto grad1 = add(mul(inode1, constant(Value<float>{0.0f})), outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);
            result[inode1] = add(result[inode1], grad1);

            break;
        }
        case TensorOp::Sub: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId inode1 = outNode.inputs_[1];

            auto grad0 = add(mul(inode0, constant(Value<float>{0.0f})), outNodeGradientId);
            auto grad1 = add(mul(inode1, constant(Value<float>{0.0f})), negate(outNodeGradientId));

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
        case TensorOp::HBroadcast: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = hReduce(outNodeGradientId);
            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::VBroadcast: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = vReduce(outNodeGradientId);
            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::HReduce: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId dimensionSizeNode = getDim(inode0, 1);

            auto grad0 =
                hBroadcast2(add(mul(inode0, constant(Value<float>{0.0f})), outNodeGradientId),
                            dimensionSizeNode);

            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::VReduce: {
            NodeId inode0 = outNode.inputs_[0];
            NodeId dimensionSizeNode = getDim(inode0, 0);

            auto grad0 =
                vBroadcast2(add(mul(inode0, constant(Value<float>{0.0f})), outNodeGradientId),
                            dimensionSizeNode);

            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::ElementwiseSquare: {
            NodeId inode0 = outNode.inputs_[0];

            auto grad0 = mul(mul(inode0, constant(Value<float>{2.0f})), outNodeGradientId);

            result[inode0] = add(result[inode0], grad0);

            break;
        }
        case TensorOp::GetDim:
        case TensorOp::Transpose:
        case TensorOp::ReluDeriv:
        case TensorOp::Negate:
            throw std::runtime_error("unimplemented");
        }
    }

    return result;
}

const std::vector<typename Graph::Node> &Graph::getNodes() const {
    return nodes_;
}

} // namespace gork
