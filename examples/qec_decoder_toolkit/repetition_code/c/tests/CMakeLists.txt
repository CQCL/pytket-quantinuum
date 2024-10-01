cmake_minimum_required(VERSION 3.14)
project(surface_code_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(CMAKE_HAVE_LIBC_PTHREAD, 0)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  test_lib
  test_lib.cc
)
target_link_libraries(
  test_lib
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_lib)
