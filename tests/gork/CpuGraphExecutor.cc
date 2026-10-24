#include "gork/Graph.hh"
#include <catch2/catch_test_macros.hpp>
#include <gork/CpuGraphExecutor.hh>
#include <vector>

TEST_CASE("it should take graphs", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};
    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    REQUIRE(&graph == executor.getGraph());
}

TEST_CASE("It should compute matmul", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};

    gork::Tensor<float> tensor1{std::vector<size_t>{1, 4}};
    gork::Tensor<float> tensor2{std::vector<size_t>{4, 2}};

    tensor1.at({0, 0}) = 1;
    tensor1.at({0, 1}) = 2;
    tensor1.at({0, 2}) = 2;
    tensor1.at({0, 3}) = 4;

    tensor2.at({0, 0}) = 0;
    tensor2.at({1, 0}) = 2;
    tensor2.at({2, 0}) = 2;
    tensor2.at({3, 0}) = 0;
    tensor2.at({0, 1}) = 0;
    tensor2.at({1, 1}) = 4;
    tensor2.at({2, 1}) = 1;
    tensor2.at({3, 1}) = 0;

    gork::NodeId node1 = graph.input();
    gork::NodeId node2 = graph.input();
    auto node3 = graph.matmul(node1, node2);

    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    std::vector<gork::Tensor<float>> inputs{tensor1, tensor2};
    executor.compute(inputs, node3);

    auto &result = executor.getValue(node3);

    REQUIRE(result.shape_[0] == 1);
    REQUIRE(result.shape_[1] == 2);

    REQUIRE(result.at({0, 0}) == 8.0);
    REQUIRE(result.at({0, 1}) == 10.0);
}

TEST_CASE("It should compute add", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};

    gork::Tensor<float> tensor1{std::vector<size_t>{2, 2}};
    gork::Tensor<float> tensor2{std::vector<size_t>{2, 2}};

    tensor1.at({0, 0}) = 1;
    tensor1.at({1, 0}) = 3;
    tensor1.at({0, 1}) = 3;
    tensor1.at({1, 1}) = 7;

    tensor2.at({0, 0}) = 0;
    tensor2.at({1, 0}) = 3;
    tensor2.at({0, 1}) = 5;
    tensor2.at({1, 1}) = 1;

    gork::NodeId node1 = graph.input();
    gork::NodeId node2 = graph.input();
    auto node3 = graph.add(node1, node2);

    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    std::vector<gork::Tensor<float>> inputs{tensor1, tensor2};

    executor.compute(inputs, node3);

    auto &result = executor.getValue(node3);

    REQUIRE(result.at({0, 0}) == 1.0);
    REQUIRE(result.at({1, 0}) == 6.0);
    REQUIRE(result.at({0, 1}) == 8.0);
    REQUIRE(result.at({1, 1}) == 8.0);
}

TEST_CASE("It should compute BroadcastAdd", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};

    gork::Tensor<float> tensor1{std::vector<size_t>{2, 2}};
    gork::Tensor<float> tensor2{std::vector<size_t>{2, 1}};

    tensor1.at({0, 0}) = 1;
    tensor1.at({1, 0}) = 3;
    tensor1.at({0, 1}) = 3;
    tensor1.at({1, 1}) = 7;

    tensor2.at({0, 0}) = 1;
    tensor2.at({1, 0}) = 5;

    gork::NodeId node1 = graph.input();
    gork::NodeId node2 = graph.input();
    auto node3 = graph.broadcastAdd(node1, node2);

    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    std::vector<gork::Tensor<float>> inputs{tensor1, tensor2};
    executor.compute(inputs, node3);

    auto &result = executor.getValue(node3);

    REQUIRE(result.at({0, 0}) == 2);
    REQUIRE(result.at({1, 0}) == 8);
    REQUIRE(result.at({0, 1}) == 4);
    REQUIRE(result.at({1, 1}) == 12);
}

TEST_CASE("It should compute Relu", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};

    gork::Tensor<float> tensor1{std::vector<size_t>{2, 3}};

    tensor1.at({0, 0}) = 6;
    tensor1.at({1, 0}) = 7;
    tensor1.at({0, 1}) = -6;
    tensor1.at({1, 1}) = -7;

    gork::NodeId node1 = graph.input();
    auto node2 = graph.relu(node1);

    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    std::vector<gork::Tensor<float>> inputs{tensor1};
    executor.compute(inputs, node2);

    auto &result = executor.getValue(node2);

    REQUIRE(result.at({0, 0}) == 6);
    REQUIRE(result.at({1, 0}) == 7);
    REQUIRE(result.at({0, 1}) == 0);
    REQUIRE(result.at({1, 1}) == 0);
}
