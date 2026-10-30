#include "gork/ops.hh"
#include "gork/Tensor.hh"
#include <catch2/catch_test_macros.hpp>

namespace gork {

TEST_CASE("Horizontal broadcast should work", "[ops]") {
    Tensor<float> tensor{{4, 1}};
    tensor.at({0, 0}) = 1.0;
    tensor.at({1, 0}) = 2.0;
    tensor.at({2, 0}) = 3.0;
    tensor.at({3, 0}) = 4.0;

    auto result = ops::hBroadcast(tensor, 3);

    REQUIRE(result.at({0, 0}) == 1.0);
    REQUIRE(result.at({1, 0}) == 2.0);
    REQUIRE(result.at({2, 0}) == 3.0);
    REQUIRE(result.at({3, 0}) == 4.0);

    REQUIRE(result.at({0, 1}) == 1.0);
    REQUIRE(result.at({1, 1}) == 2.0);
    REQUIRE(result.at({2, 1}) == 3.0);
    REQUIRE(result.at({3, 1}) == 4.0);

    REQUIRE(result.at({0, 2}) == 1.0);
    REQUIRE(result.at({1, 2}) == 2.0);
    REQUIRE(result.at({2, 2}) == 3.0);
    REQUIRE(result.at({3, 2}) == 4.0);
}

TEST_CASE("Horizontal reduce should work", "[ops]") {
    Tensor<float> tensor{{4, 3}};
    tensor.at({0, 0}) = 1.0;
    tensor.at({1, 0}) = 2.0;
    tensor.at({2, 0}) = 3.0;
    tensor.at({3, 0}) = 4.0;

    tensor.at({0, 1}) = 1.0;
    tensor.at({1, 1}) = 1.0;
    tensor.at({2, 1}) = 1.0;
    tensor.at({3, 1}) = 1.0;

    tensor.at({0, 2}) = 1.0;
    tensor.at({1, 2}) = 3.0;
    tensor.at({2, 2}) = 4.0;
    tensor.at({3, 2}) = 7.0;

    auto result = ops::hReduce(tensor);

    REQUIRE(result.at({0, 0}) == 3.0);
    REQUIRE(result.at({1, 0}) == 6.0);
    REQUIRE(result.at({2, 0}) == 8.0);
    REQUIRE(result.at({3, 0}) == 12.0);
}

TEST_CASE("Vertical broadcast should work", "[ops]") {
    Tensor<float> tensor{{1, 4}};
    tensor.at({0, 0}) = 1.0;
    tensor.at({0, 1}) = 2.0;
    tensor.at({0, 2}) = 3.0;
    tensor.at({0, 3}) = 4.0;

    auto result = ops::vBroadcast(tensor, 3);

    REQUIRE(result.at({0, 0}) == 1.0);
    REQUIRE(result.at({0, 1}) == 2.0);
    REQUIRE(result.at({0, 2}) == 3.0);
    REQUIRE(result.at({0, 3}) == 4.0);

    REQUIRE(result.at({1, 0}) == 1.0);
    REQUIRE(result.at({1, 1}) == 2.0);
    REQUIRE(result.at({1, 2}) == 3.0);
    REQUIRE(result.at({1, 3}) == 4.0);

    REQUIRE(result.at({2, 0}) == 1.0);
    REQUIRE(result.at({2, 1}) == 2.0);
    REQUIRE(result.at({2, 2}) == 3.0);
    REQUIRE(result.at({2, 3}) == 4.0);
}

TEST_CASE("Vertical reduce should work", "[ops]") {
    Tensor<float> tensor{{4, 3}};
    tensor.at({0, 0}) = 1.0;
    tensor.at({1, 0}) = 2.0;
    tensor.at({2, 0}) = 3.0;
    tensor.at({3, 0}) = 4.0;

    tensor.at({0, 1}) = 1.0;
    tensor.at({1, 1}) = 1.0;
    tensor.at({2, 1}) = 1.0;
    tensor.at({3, 1}) = 1.0;

    tensor.at({0, 2}) = 1.0;
    tensor.at({1, 2}) = 3.0;
    tensor.at({2, 2}) = 4.0;
    tensor.at({3, 2}) = 7.0;

    auto result = ops::vReduce(tensor);

    REQUIRE(result.at({0, 0}) == 10.0);
    REQUIRE(result.at({0, 1}) == 4.0);
    REQUIRE(result.at({0, 2}) == 15.0);
}

} // namespace gork
