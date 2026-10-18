#ifndef GORK_GRAPH_EXECUTOR_H_
#define GORK_GRAPH_EXECUTOR_H_

#include "gork/Graph.hh"

namespace gork {

template <typename T>
class GraphExecutor {
  public:
    GraphExecutor() = default;
    virtual ~GraphExecutor() = default;

    virtual void useGraph(Graph<T> &graph) = 0;
    virtual const Graph<T> *getGraph() const = 0;
    virtual void compute(NodeId node) = 0;
};

} // namespace gork

#endif
