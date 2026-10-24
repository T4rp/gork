#ifndef GORK_GRAPH_EXECUTOR_H_
#define GORK_GRAPH_EXECUTOR_H_

#include "gork/Graph.hh"
#include "gork/Tensor.hh"
#include <vector>

namespace gork {

template <typename T>
class GraphExecutor {
  public:
    GraphExecutor() = default;
    virtual ~GraphExecutor() = default;

    virtual void useGraph(Graph<T> &graph) = 0;
    virtual const Graph<T> *getGraph() const = 0;
    virtual Tensor<T> &getValue(NodeId nodeId) = 0;
    virtual void compute(const std::vector<Tensor<T>> &inputs, NodeId node) = 0;
};

} // namespace gork

#endif
