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
    TensorOp op;

    Tensor<T> tensor;
    Tensor<T> gradient;

    std::vector<NodeId> inputs;

    Node(Tensor<T> tensor_, TensorOp op_) : op{op_}, tensor{std::move(tensor_)} {}
};

template <typename T>
class Graph {
  public:
    std::vector<Node<T>> nodes;

    NodeId &addInput(Tensor<T> &&tensor) {
        nodes.emplace_back({std::move(tensor), TensorOp::Input});
        return nodes.size() - 1;
    }
};

} // namespace gork

#endif
