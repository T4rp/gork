add_library(mnist mnist.cc)

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
        main.cc
)

target_link_libraries(mnist_example
    PRIVATE
        mnist
)
