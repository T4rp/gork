#include "gork/Tensor.hh"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace gork {

TEST_CASE("Tensors can be made", "[Tensor]") {
    std::vector<size_t> shape{256, 128};

    Tensor<float> tensor1{shape}; // copy vector
    REQUIRE(tensor1.shape_[0] == 256);
    REQUIRE(tensor1.shape_[1] == 128);

    Tensor<float> tensor2{std::move(shape)}; // move vector
    REQUIRE(tensor2.shape_[0] == 256);
    REQUIRE(tensor2.shape_[1] == 128);

    Tensor<float> tensor3{{256, 128}}; // rvalue
    REQUIRE(tensor3.shape_[0] == 256);
    REQUIRE(tensor3.shape_[1] == 128);

    Tensor<float> tensor4{256, 128}; // list initalizer
    REQUIRE(tensor4.shape_[0] == 256);
    REQUIRE(tensor4.shape_[1] == 128);
}

TEST_CASE("Tensors values can be set", "[Tensor]") {
    std::vector<size_t> shape{256, 128};
    Tensor<float> tensor{shape};
    tensor.at({200, 100}) = 4.0;
    REQUIRE(tensor.at({200, 100}) == 4.0);
}

TEST_CASE("Tensors can be dumped", "[Tensor]") {
    std::vector<size_t> shape{2, 2};
    Tensor<float> tensor{shape};
    tensor.at({0, 0}) = 1;
    tensor.at({1, 0}) = 2;
    tensor.at({0, 1}) = 3;
    tensor.at({1, 1}) = 4;

    // swallow output
    std::stringstream buffer;
    std::streambuf *old_buffer = std::cout.rdbuf(buffer.rdbuf());

    tensor.dump_mat();

    std::cout.rdbuf(old_buffer);
}

TEST_CASE("Tensors can work as scalars", "[Tensor]") {
    Tensor<float> tensor{std::vector<size_t>{}};
    tensor.at({}) = 20.0;

    REQUIRE(tensor.at({}) == 20.0);
}

} // namespace gork
