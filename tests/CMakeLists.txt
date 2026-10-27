find_package(Catch2 3 REQUIRED)

add_executable(tests
    gork/Graph.cc
    gork/Tensor.cc
    gork/CpuGraphExecutor.cc
    gork/ops.cc
)

target_link_libraries(tests
    PRIVATE
        Catch2::Catch2WithMain
        gork
        mnist
)

target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

