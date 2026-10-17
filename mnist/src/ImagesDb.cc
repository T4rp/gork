#include "mnist/ImagesDb.h"
#include "mnist/mnist.h"

namespace mnist {

ImagesDb::ImagesDb(std::vector<uint8_t> fileData)
    : data_{std::move(fileData)} {
    count_ = readBeu32(data_.data(), 4);
    columns_ = readBeu32(data_.data(), 8);
    rows_ = readBeu32(data_.data(), 12);
}

const uint8_t *ImagesDb::getData() const {
    return data_.data() + 16;
}

const uint8_t *ImagesDb::getImage(size_t n) const {
    return getData() + columns_ * rows_ * n;
};

} // namespace mnist
