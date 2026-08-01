#include "gork/graph.hh"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Graphs can be made", "[graph]") {
    gork::Graph<float> graph{};
    REQUIRE(graph.nodes_.size() == 0);
}
