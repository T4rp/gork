#include <cstdint>
#include <vector>

#include "mnist/mnist.hh"

namespace {

uint32_t readBeu32(const uint8_t *data, size_t offset) {
    return (uint32_t(data[0 + offset]) << 24) | (uint32_t(data[1 + offset]) << 16) |
           (uint32_t(data[2 + offset]) << 8) | uint32_t(data[3 + offset]);
}

} // namespace

namespace mnist {

LabelsDb::LabelsDb(std::vector<uint8_t> fileData) : data_{std::move(fileData)} {
    count_ = readBeu32(data_.data(), 4);
    labelData_ = data_.data() + 8;
}

uint8_t LabelsDb::getLabel(size_t n) const {
    return labelData_[n];
}

ImagesDb::ImagesDb(std::vector<uint8_t> fileData) : data_{std::move(fileData)} {
    count_ = readBeu32(data_.data(), 4);
    columns_ = readBeu32(data_.data(), 8);
    rows_ = readBeu32(data_.data(), 12);

    imageData_ = data_.data() + 16;
}

const uint8_t *ImagesDb::getImage(size_t n) const {
    return imageData_ + columns_ * rows_ * n;
};

} // namespace mnist
