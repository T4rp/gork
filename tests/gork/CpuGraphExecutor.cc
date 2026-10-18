#include "gork/Graph.hh"
#include <catch2/catch_test_macros.hpp>
#include <gork/CpuGraphExecutor.hh>

TEST_CASE("it should take graphs", "[CpuGraphExecutor]") {
    gork::Graph<float> graph{};
    gork::CpuGraphExecutor<float> executor{};
    executor.useGraph(graph);

    REQUIRE(&graph == executor.getGraph());
}
