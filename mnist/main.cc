#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <zlib.h>

#include "mnist/mnist.hh"

std::vector<uint8_t> ungzFile(std::string_view path) {
    std::vector<uint8_t> unzipped{};

    unsigned char unzipBuffer[8192];

    gzFile file = gzopen(path.data(), "r");
    if (file == NULL) {
        throw std::runtime_error("Couldn't open file");
    }

    while (true) {
        size_t unzipSize = gzread(file, unzipBuffer, 8192);
        if (unzipSize > 0) {
            unzipped.insert(unzipped.end(), unzipBuffer, unzipBuffer + unzipSize);
        } else {
            break;
        }
    }

    gzclose(file);

    return unzipped;
}

char to_char(uint8_t val) {
    static const char ramp[] = " .:-=+*#%@";

    int index = (val * 9) / 255;

    return ramp[index];
};

int main() {
    gzFile file = gzopen("./mnist/assets/train-labels-idx1-ubyte.gz", "r");

    std::cout << "decompressing...\n";

    std::vector<uint8_t> labelsData = ungzFile("./mnist/assets/train-labels-idx1-ubyte.gz");
    std::vector<uint8_t> imagesData = ungzFile("./mnist/assets/train-images-idx3-ubyte.gz");

    std::cout << "decompressing done\n";

    mnist::LabelsDb labels{std::move(labelsData)};
    mnist::ImagesDb images{std::move(imagesData)};

    for (size_t i = 0; i < 10; i++) {
        std::cout << std::to_string(labels.get_label(i));

        std::cout << std::endl;

        for (size_t y = 0; y < images.columns_; y++) {
            for (size_t x = 0; x < images.rows_; x++) {
                std::cout << to_char(images.get_image(i)[x + y * images.columns_]);
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
