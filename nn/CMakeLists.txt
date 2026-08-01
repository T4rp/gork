add_executable(testing)

target_sources(testing
  PRIVATE
    main.cc
)

target_link_libraries(testing
  PRIVATE
    gork
)
