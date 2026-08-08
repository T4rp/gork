cmake_minimum_required(VERSION 3.23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Gork)

add_subdirectory(gork)
add_subdirectory(nn)
add_subdirectory(mnist)
add_subdirectory(tests)
