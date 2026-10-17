add_library(mnist
    src/mnist.cc
    src/LabelsDb.cc
    src/ImagesDb.cc
)

target_include_directories(mnist
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mnist
    PRIVATE
        z
)

add_executable(mnist_example)

target_sources(mnist_example
    PRIVATE
        example.cc
)

target_link_libraries(mnist_example
    PRIVATE
        mnist
)
