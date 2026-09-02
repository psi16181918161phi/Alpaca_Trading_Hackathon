cmake_minimum_required(VERSION 3.25)
project(xquantx_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen3 (header-only): fetched from the official upstream mirror rather
# than requiring a prior `conan install` step, so a plain
# `cmake -S src/cpp -B build/cpp` configures with no extra setup.
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

add_library(xquantx_kernel STATIC
    src/kalman_filter.cpp
)
target_include_directories(xquantx_kernel PUBLIC include)
target_link_libraries(xquantx_kernel PUBLIC Eigen3::Eigen)

option(XQUANTX_BUILD_TESTS "Build the xquantx kernel Google Test suite" ON)
if(XQUANTX_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    # Test sources live under tests/cpp/kernel per
    # alpaca_paper_trading_specifications_x_quant_x/009_xquantx_unit_tests.txt
    # section 1.9.3, outside this src/cpp/ directory.
    set(XQUANTX_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cpp/kernel")

    add_executable(xquantx_kernel_tests
        "${XQUANTX_TEST_DIR}/test_kalman_filter.cpp"
        "${XQUANTX_TEST_DIR}/test_covariance.cpp"
    )
    target_link_libraries(xquantx_kernel_tests PRIVATE xquantx_kernel gtest gtest_main)
    gtest_discover_tests(xquantx_kernel_tests)
endif()
