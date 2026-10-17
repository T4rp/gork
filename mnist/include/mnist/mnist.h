#ifndef MNIST_H_
#define MNIST_H_

#include <cstddef>
#include <cstdint>

namespace mnist {

uint32_t readBeu32(const uint8_t *data, size_t offset);

} // namespace mnist

#endif
