add_executable(mnist)

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
