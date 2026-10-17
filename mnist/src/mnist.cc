#include "mnist/mnist.h"
#include <cstddef>
#include <cstdint>

uint32_t mnist::readBeu32(const uint8_t *data, size_t offset) {
    return (uint32_t(data[0 + offset]) << 24) | (uint32_t(data[1 + offset]) << 16) |
           (uint32_t(data[2 + offset]) << 8) | uint32_t(data[3 + offset]);
}
