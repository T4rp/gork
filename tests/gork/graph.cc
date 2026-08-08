#include "gork/graph.hh"
#include "gork/tensor.hh"
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("Graphs can be made", "[graph]") {
    gork::Graph<float> graph{};
    REQUIRE(graph.nodes_.size() == 0);
}

TEST_CASE("Nodes can be inserted", "[graph]") {
    gork::Graph<float> graph{};

    graph.addInput(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    REQUIRE(graph.nodes_.size() == 1);
}

TEST_CASE("Nodes can be multiplied", "[graph]") {
    gork::Graph<float> graph{};

    gork::NodeId node1 = graph.addInput(gork::Tensor<float>{std::vector<size_t>{4, 4}});
    gork::NodeId node2 = graph.addInput(gork::Tensor<float>{std::vector<size_t>{4, 3}});

    graph.matmul(node1, node2);

    REQUIRE(graph.nodes_.size() == 3);
}
