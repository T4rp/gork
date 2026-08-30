#ifndef MNIST_H_
#define MNIST_H_

#include <cstdint>
#include <vector>

namespace mnist {

class LabelsDb {
  private:
    std::vector<uint8_t> data_;

  public:
    uint32_t count_;
    uint8_t *labelData_;

    LabelsDb(std::vector<uint8_t> fileData);
    uint8_t get_label(size_t n) const;
};

class ImagesDb {
  private:
    std::vector<uint8_t> data_;

  public:
    uint32_t count_;
    uint32_t rows_;
    uint32_t columns_;
    uint8_t *imageData_;

    ImagesDb(std::vector<uint8_t> fileData);
    const uint8_t *get_image(size_t n) const;
};

} // namespace mnist

#endif
