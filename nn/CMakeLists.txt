add_executable(testing)

target_sources(testing
    PRIVATE
        src/main.cc
)

target_link_libraries(testing
    PRIVATE
        gork
)
