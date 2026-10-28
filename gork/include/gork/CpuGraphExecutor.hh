#ifndef GORK_CPU_GRAPH_EXECUTOR_H_
#define GORK_CPU_GRAPH_EXECUTOR_H_

#include "gork/Graph.hh"
#include "gork/GraphExecutor.hh"
#include "gork/Tensor.hh"
#include "gork/ops.hh"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gork {

template <typename T>
class CpuGraphExecutor : public GraphExecutor<T> {
  public:
    CpuGraphExecutor();
    ~CpuGraphExecutor();

    void useGraph(Graph<T> &graph);
    const Graph<T> *getGraph() const;
    Tensor<T> &getValue(NodeId nodeId);

    void compute(const std::vector<Tensor<T>> &inputs, NodeId node);

  private:
    Graph<T> *graph_;
    std::vector<Tensor<T>> tensors_;
};

template <typename T>
CpuGraphExecutor<T>::CpuGraphExecutor()
    : graph_{nullptr},
      tensors_{} {}

template <typename T>
CpuGraphExecutor<T>::~CpuGraphExecutor() {
    graph_ = nullptr;
}

template <typename T>
void CpuGraphExecutor<T>::useGraph(Graph<T> &graph) {
    graph_ = &graph;
    tensors_ = std::vector<Tensor<T>>{};

    auto &nodes = graph_->getNodes();
    size_t nodeCount = nodes.size();

    for (size_t i = 0; i < nodeCount; i++) {
        auto node = graph_->get(i);
        if (node.op_ == TensorOp::Constant) {
            tensors_.push_back(node.value_.asTensor());
        } else {
            tensors_.push_back(Tensor<T>{});
        }
    }
}

template <typename T>
const Graph<T> *CpuGraphExecutor<T>::getGraph() const {
    return graph_;
}

template <typename T>
Tensor<T> &CpuGraphExecutor<T>::getValue(NodeId nodeId) {
    return tensors_[nodeId];
}

template <typename T>
void CpuGraphExecutor<T>::compute(const std::vector<Tensor<T>> &inputs, NodeId nodeId) {
    if (graph_ == nullptr) {
        throw std::runtime_error("Graph not present");
    }

    for (size_t i = 0; i < graph_->inputs_.size(); i++) {
        NodeId nodeId = graph_->inputs_[i];
        tensors_[nodeId] = inputs[i];
    }

    std::vector<NodeId> dependencies{};
    std::vector<NodeId> stack{};

    stack.push_back(nodeId);

    while (stack.size() > 0) {
        NodeId nodeId = stack.back();
        stack.pop_back();

        dependencies.push_back(nodeId);

        auto &node = graph_->get(nodeId);

        for (NodeId parentId : node.inputs_) {
            stack.push_back(parentId);
        }
    }

    for (size_t i = dependencies.size(); i-- > 0;) {
        NodeId nodeId = dependencies[i];
        auto &node = graph_->get(nodeId);
        auto *tensor = &tensors_[nodeId];

        switch (node.op_) {
        case TensorOp::Input:
        case TensorOp::Constant:
            break;
        case TensorOp::NoOp:
        case TensorOp::Output: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = child0;
            break;
        }
        case TensorOp::Matmul: {
            auto &child0 = tensors_[node.inputs_[0]];
            auto &child1 = tensors_[node.inputs_[1]];
            *tensor = gork::ops::matmul(child0, child1);
            break;
        }
        case TensorOp::Transpose: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::transpose(child0);
            break;
        }
        case TensorOp::Add: {
            auto &child0 = tensors_[node.inputs_[0]];
            auto &child1 = tensors_[node.inputs_[1]];
            *tensor = gork::ops::add(child0, child1);
            break;
        }
        case TensorOp::Sub: {
            auto &child0 = tensors_[node.inputs_[0]];
            auto &child1 = tensors_[node.inputs_[1]];
            *tensor = gork::ops::sub(child0, child1);
            break;
        }
        case TensorOp::Mul: {
            auto &child0 = tensors_[node.inputs_[0]];
            auto &child1 = tensors_[node.inputs_[1]];
            *tensor = gork::ops::mul(child0, child1);
            break;
        }
        case TensorOp::ElementwiseSquare: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::elementwiseSquare(child0);
            break;
        }
        case TensorOp::Relu: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::relu(child0);
            break;
        }
        case TensorOp::ReluDeriv: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::reluDeriv(child0);
            break;
        }
        case TensorOp::Negate: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::negate(child0);
            break;
        }
        case TensorOp::HBroadcast: {
            auto &child0 = tensors_[node.inputs_[0]];

            auto countNodeId = node.inputs_[1];
            auto &node = graph_->get(countNodeId);

            size_t count = std::get<size_t>(node.value_.val_);
            *tensor = gork::ops::hBroadcast(child0, count);

            break;
        }
        case TensorOp::HReduce: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::hReduce(child0);
            break;
        }
        case TensorOp::VBroadcast: {
            auto &child0 = tensors_[node.inputs_[0]];

            auto countNodeId = node.inputs_[1];
            auto &node = graph_->get(countNodeId);

            size_t count = std::get<size_t>(node.value_.val_);
            *tensor = gork::ops::vBroadcast(child0, count);

            break;
        }
        case TensorOp::VReduce: {
            auto &child0 = tensors_[node.inputs_[0]];
            *tensor = gork::ops::vReduce(child0);
            break;
        }
        }
    }
}

} // namespace gork

#endif
