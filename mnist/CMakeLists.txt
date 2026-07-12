add_executable(mnist)

target_include_directories(mnist
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(mnist
  PRIVATE
    main.cc
    mnist.cc
)

target_link_libraries(mnist
  PRIVATE
    gork
	z
)
