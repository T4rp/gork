#include "gork/Graph.hh"
#include "gork/ops.hh"
#include <catch2/catch_test_macros.hpp>
#include <vector>

namespace gork {

TEST_CASE("Graphs can be made", "[Graph]") {
    Graph graph{};
    REQUIRE(graph.getNodes().size() == 0);
}

TEST_CASE("Nodes can be inserted", "[Graph]") {
    Graph graph{};

    graph.input();
    REQUIRE(graph.getNodes().size() == 1);
}

TEST_CASE("Graph inputs are recorded", "[Graph]") {
    Graph graph{};

    auto nodeId = graph.input();

    REQUIRE(nodeId == graph.inputs_[0]);
}

TEST_CASE("Graph outputs are recorded", "[Graph]") {
    Graph graph{};

    auto inputId = graph.input();
    auto outputId = graph.output(inputId);

    REQUIRE(outputId == graph.outputs_[0]);
}

TEST_CASE("Nodes can be multiplied", "[Graph]") {
    Graph graph{};

    NodeId node1 = graph.input();
    NodeId node2 = graph.input();

    graph.matmul(node1, node2);

    REQUIRE(graph.getNodes().size() == 3);
}

TEST_CASE("Nodes can be added", "[Graph]") {
    Graph graph{};

    NodeId node1 = graph.input();
    NodeId node2 = graph.input();

    graph.add(node1, node2);

    REQUIRE(graph.getNodes().size() == 3);
}

TEST_CASE("Nodes can be retrieved", "[Graph]") {
    Graph graph{};

    NodeId node1 = graph.input();
    Graph::Node &node = graph.get(node1);

    REQUIRE(node.op_ == TensorOp::Input);
}

TEST_CASE("Graph can build backwards pass", "[Graph]") {
    Graph graph{};

    auto node1 = graph.input();
    auto node2 = graph.input();
    auto node3 = graph.matmul(node1, node2);

    auto gradientMap = graph.mapGradientNodes(node3);

    // TODO: add test cases
}

} // namespace gork
