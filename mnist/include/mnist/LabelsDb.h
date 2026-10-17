#ifndef MNIST_LABELS_DB_H_
#define MNIST_LABELS_DB_H_

#include <cstdint>
#include <vector>

namespace mnist {

class LabelsDb {
  private:
    std::vector<uint8_t> data_;

  public:
    uint32_t count_;

    LabelsDb(std::vector<uint8_t> fileData);
    const uint8_t *getData() const;
    uint8_t getLabel(size_t n) const;
};

} // namespace mnist

#endif
