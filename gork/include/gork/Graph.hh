#ifndef GORK_GRAPH_H_
#define GORK_GRAPH_H_

#include "gork/Value.hh"
#include "gork/ops.hh"
#include <unordered_map>
#include <vector>

namespace gork {

using NodeId = size_t;

class Graph {
  public:
    class Node {
      public:
        TensorOp op_;
        size_t paramIndex_;
        std::vector<NodeId> inputs_;

        Value<float> value_;

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
    NodeId constant(Value<float> value);
    NodeId matmul(NodeId lhs, NodeId rhs);
    NodeId transpose(NodeId tensor);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId sub(NodeId lhs, NodeId rhs);
    NodeId mul(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId tensor);
    NodeId reluDeriv(NodeId tensor);
    NodeId negate(NodeId nodeId);
    NodeId hBroadcast(NodeId nodeId, size_t count);
    NodeId hReduce(NodeId nodeId);
    NodeId vBroadcast(NodeId nodeId, size_t count);
    NodeId vReduce(NodeId nodeId);

    std::unordered_map<NodeId, NodeId> mapGradientNodes(NodeId toCompute);

  private:
    std::vector<Node> nodes_;
};

} // namespace gork

#endif
