#ifndef GORK_CPU_GRAPH_EXECUTOR_H_
#define GORK_CPU_GRAPH_EXECUTOR_H_

#include "gork/Graph.hh"
#include "gork/GraphExecutor.hh"

namespace gork {

template <typename T>
class CpuGraphExecutor : public GraphExecutor<T> {
  public:
    CpuGraphExecutor();

    void useGraph(const Graph<T> &graph);
    const Graph<T> *getGraph() const;
    void compute(NodeId node);

  private:
    const Graph<T> *graph_;
};

template <typename T>
CpuGraphExecutor<T>::CpuGraphExecutor() {}

template <typename T>
void CpuGraphExecutor<T>::useGraph(const Graph<T> &graph) {
    graph_ = &graph;
}

template <typename T>
const Graph<T> *CpuGraphExecutor<T>::getGraph() const {
    return graph_;
}

template <typename T>
void CpuGraphExecutor<T>::compute(NodeId node) {}

} // namespace gork

#endif
