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

    Node() : op{TensorOp::NoOp} {}
};

template <typename T>
class Graph {
  public:
    std::vector<Node<T>> nodes;
};

} // namespace gork

#endif
