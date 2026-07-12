#ifndef MNIST_H_
#define MNIST_H_

#include <cstdint>
#include <vector>

namespace mnist {

class LabelsDb {
  private:
    std::vector<uint8_t> data;

  public:
    uint32_t count;
    uint8_t *label_data;

    LabelsDb(std::vector<uint8_t> fileData);
    uint8_t get_label(size_t n) const;
};

class ImagesDb {
  private:
    std::vector<uint8_t> data;

  public:
    uint32_t count;
    uint32_t rows;
    uint32_t columns;
    uint8_t *image_data;

    ImagesDb(std::vector<uint8_t> fileData);
    const uint8_t *get_image(size_t n) const;
};

} // namespace mnist

#endif
