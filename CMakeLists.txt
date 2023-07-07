cmake_minimum_required(VERSION 3.20.0)
project(mlidk)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(src/CMakeLists.txt)
include(unittests/CMakeLists.txt)

# -------------------------------
# Library setup
# -------------------------------
find_package(LLVM 16.0.6 REQUIRED CONFIG EXACT)


message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR 16.0.6 REQUIRED CONFIG EXACT)

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Download Gtest
include(FetchContent)

# Source: https://google.github.io/googletest/quickstart-cmake.html
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)


# -------------------------------
# Executable and Flags
# -------------------------------
add_executable(mlidk ${ALL_SRC_FILES})

# Add headers
target_include_directories(mlidk PRIVATE "src/include")

# Debug Build flags
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
  -fsanitize=address \
  -fno-omit-frame-pointer \
  -fsanitize=undefined \
  -fno-optimize-sibling-calls \
  -g"
)
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} \
  -fsanitize=address,undefined \
  -g"
)

# General Build flags
target_compile_options(mlidk PRIVATE
  -Wall
  -Werror
  -Wpedantic
  -Wextra
  -fPIE
)


# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link against LLVM libraries
target_link_libraries(mlidk ${llvm_libs})


# -------------------------------
# Google Test targets
# -------------------------------

enable_testing()

add_executable(all_tests
  ${TEST_FILES}
  ${SRC_FILES}
  ${HPP_FILES}
)

target_compile_options(all_tests PRIVATE
  -Wall
  -Werror
  -Wpedantic
  -Wextra
  -fPIE
)

target_include_directories(all_tests PRIVATE "src/include")

target_link_libraries(all_tests
  GTest::gtest_main
  ${llvm_libs}
)


include(GoogleTest)
gtest_discover_tests(all_tests)

