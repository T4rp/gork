#include <iostream>
#include <random>
#include <vector>

#include "gork/CpuGraphExecutor.hh"
#include "gork/Graph.hh"
#include "gork/Tensor.hh"
#include "gork/Value.hh"
#include "gork/ops.hh"

const size_t kBatchSize = 16;

int main() {
    std::random_device rd;
    std::mt19937 rngGen(rd());

    gork::Graph graph{};

    auto input = graph.input();
    auto expected = graph.input();
    auto weights1 = graph.input();
    auto bias1 = graph.input();
    auto weights2 = graph.input();
    auto bias2 = graph.input();

    auto bias1Broad = graph.hBroadcast(bias1, kBatchSize);
    auto bias2Broad = graph.hBroadcast(bias2, kBatchSize);

    auto layer1 = graph.relu(graph.add(graph.matmul(weights1, input), bias1Broad));
    auto output = graph.relu(graph.add(graph.matmul(weights2, layer1), bias2Broad));

    auto loss = graph.sub(output, expected);
    loss = graph.elemSquare(loss);
    loss = graph.hReduce(loss);
    loss = graph.vReduce(loss);
    loss =
        graph.mul(loss, graph.constant(gork::Value<float>{1.0f / static_cast<float>(kBatchSize)}));

    auto lossGradientMap = graph.mapGradientNodes(loss);

    gork::CpuGraphExecutor<float> executor{};
    // executor.loggingEnabled_ = true;
    executor.useGraph(graph);

    std::vector<gork::Tensor<float>> inputs{
        {2, kBatchSize}, // input
        {1, kBatchSize}, // expected
        {2, 2},          // weights1
        {2, 1},          // bias1
        {1, 2},          // weights2
        {1, 1},          // bias2
    };

    for (auto &tensor : inputs) {
        tensor.fill(0.0);
    }

    inputs[2].fillRandHe(rngGen);
    inputs[4].fillRandHe(rngGen);

    std::uniform_int_distribution<> distrib(0, 1);

    for (size_t i = 0; i < 10000; i++) {
        for (size_t i = 0; i < kBatchSize; i++) {
            int i1 = distrib(rngGen);
            int i2 = distrib(rngGen);
            int o1 = i1 ^ i2;

            inputs[0].at({0, i}) = i1;
            inputs[0].at({1, i}) = i2;

            inputs[1].at({0, i}) = o1;
        }

        executor.compute(inputs, loss);

        auto inputTensor = executor.getValue(input);
        auto expectedTensor = executor.getValue(expected);
        auto resultTensor = executor.getValue(output);
        auto lossValue = executor.getValue(loss);

        executor.compute(inputs, lossGradientMap[weights1]);
        auto w1GradTensor = executor.getValue(lossGradientMap[weights1]);

        executor.compute(inputs, lossGradientMap[bias1]);
        auto b1GradTensor = executor.getValue(lossGradientMap[bias1]);

        executor.compute(inputs, lossGradientMap[weights2]);
        auto w2GradTensor = executor.getValue(lossGradientMap[weights2]);

        executor.compute(inputs, lossGradientMap[bias2]);
        auto b2GradTensor = executor.getValue(lossGradientMap[bias2]);

        std::cout << "-----------------------------------\n";
        std::cout << "input: " << inputTensor << std::endl;
        std::cout << "expected: " << expectedTensor << std::endl;
        std::cout << "result: " << resultTensor << std::endl;
        std::cout << "MSE loss average: " << lossValue << std::endl;

        std::cout << "weight1 grad " << lossGradientMap[weights1] << ": " << w1GradTensor
                  << std::endl;
        std::cout << "bias1 grad: " << b1GradTensor << std::endl;
        std::cout << "weight2 grad: " << w2GradTensor << std::endl;
        std::cout << "bias2 grad: " << b2GradTensor << std::endl;

        if (lossValue.data_[0] < 0.05) {
            std::cout << "Reached a loss less that 0.05. Stopping training :)\n";
            break;
        }

        float lr = 0.01;

        inputs[2] = gork::ops::sub(inputs[2],
                                   gork::ops::mul(w1GradTensor, gork::Value<float>{lr}.asTensor()));

        inputs[3] = gork::ops::sub(inputs[3],
                                   gork::ops::mul(b1GradTensor, gork::Value<float>{lr}.asTensor()));

        inputs[4] = gork::ops::sub(inputs[4],
                                   gork::ops::mul(w2GradTensor, gork::Value<float>{lr}.asTensor()));

        inputs[5] = gork::ops::sub(inputs[5],
                                   gork::ops::mul(b2GradTensor, gork::Value<float>{lr}.asTensor()));
    }

    std::cout << "\nFinal values:\n";

    std::cout << "Weights 1: " << inputs[2] << std::endl;
    std::cout << "Bias 1: " << inputs[3] << std::endl;
    std::cout << "Weights 2: " << inputs[4] << std::endl;
    std::cout << "Bias 2: " << inputs[5] << std::endl;

    return 0;
}
