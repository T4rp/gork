#include "gork/tensor.hh"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <sstream>

TEST_CASE("Tensors can be made", "[tensor]") {
    std::vector<size_t> shape{256, 128};
    gork::Tensor<float> tensor{shape};
}

TEST_CASE("Tensors values can be set", "[tensor]") {
    std::vector<size_t> shape{256, 128};
    gork::Tensor<float> tensor{shape};
    tensor.at({100, 200}) = 4.0;
    REQUIRE(tensor.at({100, 200}) == 4.0);
}

TEST_CASE("Tensors can be dumped", "[tensor]") {
    std::vector<size_t> shape{2, 2};
    gork::Tensor<float> tensor{shape};
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
