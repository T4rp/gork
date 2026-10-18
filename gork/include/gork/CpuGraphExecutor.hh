#ifndef GORK_CPU_GRAPH_EXECUTOR_H_
#define GORK_CPU_GRAPH_EXECUTOR_H_

#include "gork/Graph.hh"
#include "gork/GraphExecutor.hh"
#include "gork/ops.hh"
#include <stdexcept>

namespace gork {

template <typename T>
class CpuGraphExecutor : public GraphExecutor<T> {
  public:
    CpuGraphExecutor();

    void useGraph(Graph<T> &graph);
    const Graph<T> *getGraph() const;

    void compute(NodeId node);

  private:
    Graph<T> *graph_;
};

template <typename T>
CpuGraphExecutor<T>::CpuGraphExecutor() {}

template <typename T>
void CpuGraphExecutor<T>::useGraph(Graph<T> &graph) {
    graph_ = &graph;
}

template <typename T>
const Graph<T> *CpuGraphExecutor<T>::getGraph() const {
    return graph_;
}

template <typename T>
void CpuGraphExecutor<T>::compute(NodeId lastNodeId) {
    if (graph_ == nullptr) {
        throw std::runtime_error("Graph not present");
    }

    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(lastNodeId);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        auto &node = graph_->get(nodeId);

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    std::reverse(std::begin(dependencies), std::end(dependencies));

    for (size_t i = dependencies.size(); i--;) {
        NodeId nodeId = dependencies[i];
        auto &node = graph_->get(nodeId);

        switch (node.op_) {
        case TensorOp::Matmul: {
            auto &child0 = graph_->get(node.inputs_[0]);
            auto &child1 = graph_->get(node.inputs_[1]);
            node.tensor_ = gork::matmul(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Transpose: {
            auto &child0 = graph_->get(node.inputs_[0]);
            node.tensor_ = gork::transpose(child0.tensor_);
            break;
        }
        case TensorOp::Add: {
            auto &child0 = graph_->get(node.inputs_[0]);
            auto &child1 = graph_->get(node.inputs_[1]);
            node.tensor_ = gork::add(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Sub: {
            auto &child0 = graph_->get(node.inputs_[0]);
            auto &child1 = graph_->get(node.inputs_[1]);
            node.tensor_ = gork::sub(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Mul: {
            auto &child0 = graph_->get(node.inputs_[0]);
            auto &child1 = graph_->get(node.inputs_[1]);
            node.tensor_ = gork::mul(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::ElementwiseSquare: {
            auto &child0 = graph_->get(node.inputs_[0]);
            node.tensor_ = gork::elementwiseSquare(child0.tensor_);
            break;
        }
        case TensorOp::BroadcastAdd: {
            auto &child0 = graph_->get(node.inputs_[0]);
            auto &child1 = graph_->get(node.inputs_[1]);
            node.tensor_ = gork::broadcastAdd(child0.tensor_, child1.tensor_);
            break;
        }
        case TensorOp::Relu: {
            auto &child0 = graph_->get(node.inputs_[0]);
            node.tensor_ = gork::relu(child0.tensor_);
            break;
        }
        case TensorOp::ReluDeriv: {
            auto &child0 = graph_->get(node.inputs_[0]);
            node.tensor_ = gork::reluDeriv(child0.tensor_);
            break;
        }
        }
    }
}

} // namespace gork

#endif
