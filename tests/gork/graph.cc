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
