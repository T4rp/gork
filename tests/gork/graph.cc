#include "gork/graph.hh"
#include "gork/tensor.hh"
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("Graphs can be made", "[graph]") {
    gork::Graph<float> graph{};
    REQUIRE(graph.testNodes().size() == 0);
}

TEST_CASE("Nodes can be inserted", "[graph]") {
    gork::Graph<float> graph{};

    graph.addInput(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    REQUIRE(graph.testNodes().size() == 1);
}

TEST_CASE("Nodes can be multiplied", "[graph]") {
    gork::Graph<float> graph{};

    gork::NodeId node1 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    gork::NodeId node2 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 3}});

    graph.matmul(node1, node2);

    REQUIRE(graph.testNodes().size() == 3);
}

TEST_CASE("Nodes can be added", "[graph]") {
    gork::Graph<float> graph{};

    gork::NodeId node1 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    gork::NodeId node2 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 4}});

    graph.add(node1, node2);

    REQUIRE(graph.testNodes().size() == 3);
}

TEST_CASE("Nodes can be retrieved", "[graph]") {
    gork::Graph<float> graph{};

    gork::NodeId node1 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 2}});
    gork::Node<float> &node = graph.get(node1);

    REQUIRE(node.tensor_.shape_[0] == 4);
    REQUIRE(node.tensor_.shape_[1] == 2);
}

TEST_CASE("Nodes can be computed", "[graph]") {
    gork::Graph<float> graph{};

    gork::NodeId node1 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    gork::NodeId node2 = graph.addTensor(gork::Tensor<float>{std::vector<size_t>{4, 4}});

    auto node3 = graph.add(node1, node2);
    graph.compute(node3);
}

TEST_CASE("Matmul should compute", "[graph]") {
    gork::Graph<float> graph{};

    gork::Tensor<float> tensor1{std::vector<size_t>{3, 4}};
    gork::Tensor<float> tensor2{std::vector<size_t>{4, 5}};

    tensor1.at({0, 0}) = 1;
    tensor1.at({1, 1}) = 2;
    tensor1.at({0, 2}) = 2;
    tensor1.at({2, 3}) = 4;

    tensor2.at({0, 0}) = 1;
    tensor2.at({1, 1}) = 1;
    tensor2.at({2, 2}) = 1;
    tensor2.at({1, 0}) = 2;
    tensor2.at({2, 0}) = 2;
    tensor2.at({0, 3}) = 4;

    gork::NodeId node1 = graph.addTensor(std::move(tensor1));
    gork::NodeId node2 = graph.addTensor(std::move(tensor2));

    auto node3 = graph.matmul(node1, node2);
    gork::Tensor<float> &result = graph.compute(node3);

    result.dump_mat();

    REQUIRE(result.shape_[0] == 3);
    REQUIRE(result.shape_[1] == 5);
}
