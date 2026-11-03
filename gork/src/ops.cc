#include "gork/ops.hh"

namespace gork {

const char *toString(TensorOp op) {
    switch (op) {
    case TensorOp::NoOp:
        return "NoOp";
    case TensorOp::Input:
        return "Input";
    case TensorOp::Output:
        return "Output";
    case TensorOp::Constant:
        return "Constant";
    case TensorOp::Matmul:
        return "Matmul";
    case TensorOp::Transpose:
        return "Transpose";
    case TensorOp::Add:
        return "Add";
    case TensorOp::Sub:
        return "Sub";
    case TensorOp::Mul:
        return "Mul";
    case TensorOp::ElementwiseSquare:
        return "ElementwiseSquare";
    case TensorOp::Relu:
        return "Relu";
    case TensorOp::ReluDeriv:
        return "ReluDeriv";
    case TensorOp::Negate:
        return "Negate";
    case TensorOp::HBroadcast:
        return "HBroadcast";
    case TensorOp::HReduce:
        return "HReduce";
    case TensorOp::VBroadcast:
        return "VBroadcast";
    case TensorOp::VReduce:
        return "VReduce";
    case TensorOp::GetDim:
        return "GetDim";
    default:
        return "Unknown";
    }
}

}; // namespace gork
