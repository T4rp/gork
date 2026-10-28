#include "gork/Graph.hh"
#include "gork/ops.hh"
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("Graphs can be made", "[Graph]") {
    gork::Graph graph{};
    REQUIRE(graph.getNodes().size() == 0);
}

TEST_CASE("Nodes can be inserted", "[Graph]") {
    gork::Graph graph{};

    graph.input();
    REQUIRE(graph.getNodes().size() == 1);
}

TEST_CASE("Graph inputs are recorded", "[Graph]") {
    gork::Graph graph{};

    auto nodeId = graph.input();

    REQUIRE(nodeId == graph.inputs_[0]);
}

TEST_CASE("Graph outputs are recorded", "[Graph]") {
    gork::Graph graph{};

    auto inputId = graph.input();
    auto outputId = graph.output(inputId);

    REQUIRE(outputId == graph.outputs_[0]);
}

TEST_CASE("Nodes can be multiplied", "[Graph]") {
    gork::Graph graph{};

    gork::NodeId node1 = graph.input();
    gork::NodeId node2 = graph.input();

    graph.matmul(node1, node2);

    REQUIRE(graph.getNodes().size() == 3);
}

TEST_CASE("Nodes can be added", "[Graph]") {
    gork::Graph graph{};

    gork::NodeId node1 = graph.input();
    gork::NodeId node2 = graph.input();

    graph.add(node1, node2);

    REQUIRE(graph.getNodes().size() == 3);
}

TEST_CASE("Nodes can be retrieved", "[Graph]") {
    gork::Graph graph{};

    gork::NodeId node1 = graph.input();
    gork::Graph::Node &node = graph.get(node1);

    REQUIRE(node.op_ == gork::TensorOp::Input);
}

TEST_CASE("Graph can build backwards pass", "[Graph]") {
    gork::Graph graph{};

    auto node1 = graph.input();
    auto node2 = graph.input();
    auto node3 = graph.matmul(node1, node2);

    auto gradientMap = graph.mapGradientNodes(node3);

    // TODO: add test cases
}
