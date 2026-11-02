#include <iostream>
#include <vector>

#include "gork/CpuGraphExecutor.hh"
#include "gork/Graph.hh"
#include "gork/Tensor.hh"
#include "gork/Value.hh"

const size_t kBatchSize = 16;

int main() {
    gork::Graph graph{};

    auto input = graph.input();
    auto expected = graph.input();
    auto weights1 = graph.input();
    auto bias1 = graph.input();
    auto weights2 = graph.input();
    auto bias2 = graph.input();

    bias1 = graph.hBroadcast(bias1, kBatchSize);
    bias2 = graph.hBroadcast(bias2, kBatchSize);

    auto layer1 = graph.relu(graph.add(graph.matmul(weights1, input), bias1));
    auto output = graph.relu(graph.add(graph.matmul(weights2, layer1), bias2));

    auto loss = graph.sub(output, expected);
    loss = graph.elemSquare(loss);
    loss = graph.hReduce(loss);
    loss = graph.vReduce(loss);
    loss = graph.mul(loss,
                     graph.constant(gork::Value<float>{1.0f / static_cast<float>(kBatchSize * 2)}));

    auto lossGradientMap = graph.mapGradientNodes(loss);

    // TODO: Fix gradient calc here
    auto inputGradient = lossGradientMap[layer1];

    std::vector<gork::Tensor<float>> inputs{
        {2, kBatchSize}, // input
        {2, kBatchSize}, // expected
        {2, 2},          // weights1
        {2, 1},          // bias1
        {2, 2},          // weights2
        {2, 1},          // bias2
    };

    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    executor.compute(inputs, loss);
    executor.compute(inputs, inputGradient);

    std::cout << "hello world\n";
    return 0;
}
