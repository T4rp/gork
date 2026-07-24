#include <cstdint>
#include <vector>

#include "gork/mnist.hh"

namespace {

uint32_t read_beu32(const uint8_t *data, size_t offset) {
    return (uint32_t(data[0 + offset]) << 24) | (uint32_t(data[1 + offset]) << 16) |
           (uint32_t(data[2 + offset]) << 8) | uint32_t(data[3 + offset]);
}

} // namespace

namespace mnist {

LabelsDb::LabelsDb(std::vector<uint8_t> fileData) : data{std::move(fileData)} {
    count = read_beu32(data.data(), 4);

    label_data = data.data() + 8;
}

uint8_t LabelsDb::get_label(size_t n) const {
    return label_data[n];
}

ImagesDb::ImagesDb(std::vector<uint8_t> fileData) : data{std::move(fileData)} {
    count = read_beu32(data.data(), 4);
    columns = read_beu32(data.data(), 8);
    rows = read_beu32(data.data(), 12);

    image_data = data.data() + 16;
}

const uint8_t *ImagesDb::get_image(size_t n) const {
    return image_data + columns * rows * n;
};

} // namespace mnist
