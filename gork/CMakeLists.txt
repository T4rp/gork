add_library(gork
    src/Graph.cc
    src/ops.cc
)

target_include_directories(gork PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

