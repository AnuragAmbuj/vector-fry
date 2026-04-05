include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(fry_tests
    test_vector_store.cpp
    test_distance.cpp
    test_flat_index.cpp
    test_distance_simd.cpp
    test_hnsw.cpp
)
target_link_libraries(fry_tests PRIVATE fry_vector_lib Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(fry_tests)
