#ifndef MNIST_IMAGES_DB_H_
#define MNIST_IMAGES_DB_H_

#include <cstdint>
#include <vector>

namespace mnist {

class ImagesDb {
  private:
    std::vector<uint8_t> data_;

  public:
    uint32_t count_;
    uint32_t rows_;
    uint32_t columns_;

    ImagesDb(std::vector<uint8_t> fileData);
    const uint8_t *getData() const;
    const uint8_t *getImage(size_t n) const;
};

} // namespace mnist

#endif
