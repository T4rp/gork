#include "mnist/LabelsDb.h"
#include "mnist/mnist.h"
#include <cstdint>

namespace mnist {

LabelsDb::LabelsDb(std::vector<uint8_t> fileData)
    : data_{std::move(fileData)} {
    count_ = readBeu32(data_.data(), 4);
}

const uint8_t *LabelsDb::getData() const {
    return data_.data() + 8;
}

uint8_t LabelsDb::getLabel(size_t n) const {
    return getData()[n];
}

} // namespace mnist
