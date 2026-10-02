cmake_minimum_required(VERSION 3.20)

project(
  kernelwrappers
  VERSION 0.1
  LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(KERNELWRAPPERS_ROOT "${CMAKE_CURRENT_LIST_DIR}")

set(KERNELWRAPPERS_CUDAWRAPPERS_VERSION
    "1.0.0"
    CACHE STRING "cudawrappers version to fetch")

if(NOT TARGET cudawrappers::cu)
  set(CUDAWRAPPERS_COMPONENTS cu cufft nvrtc nvtx)
  FetchContent_Declare(
    cudawrappers
    GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
    GIT_TAG ${KERNELWRAPPERS_CUDAWRAPPERS_VERSION})
  FetchContent_MakeAvailable(cudawrappers)
endif()

option(KERNELWRAPPERS_BUILD_EXAMPLES "Build the examples" OFF)
option(KERNELWRAPPERS_BUILD_TESTS "Build the tests" OFF)

if(DEFINED ENV{HOME})
  set(KERNELWRAPPERS_CACHE_PATH
      "$ENV{HOME}/.kernelwrappers/KernelCache"
      CACHE STRING "Location to cache compiled kernels")
else()
  set(KERNELWRAPPERS_CACHE_PATH
      "/tmp/kernelwrappers/KernelCache"
      CACHE STRING "Location to cache compiled kernels")
endif()

message(STATUS "kernelwrappers cache location: ${KERNELWRAPPERS_CACHE_PATH}")

set(_kernelwrappers_sources
    src/Kernel.cc src/KernelCache.cc src/KernelConfig.cc src/JITKernel.cc
    src/KernelCompilationQueue.cc src/Log.cc)

add_library(kernelwrappers SHARED ${_kernelwrappers_sources})
add_library(kernelwrappers::kernelwrappers ALIAS kernelwrappers)

target_include_directories(
  kernelwrappers PUBLIC "$<BUILD_INTERFACE:${KERNELWRAPPERS_ROOT}/include>"
                        "$<INSTALL_INTERFACE:include>")

target_link_libraries(kernelwrappers PUBLIC cudawrappers::cu)
target_link_libraries(kernelwrappers PUBLIC cudawrappers::nvrtc)
target_link_libraries(kernelwrappers PUBLIC OpenSSL::Crypto)

target_compile_definitions(
  kernelwrappers
  PRIVATE KERNELWRAPPERS_CACHE_PATH="${KERNELWRAPPERS_CACHE_PATH}")

if(KERNELWRAPPERS_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(KERNELWRAPPERS_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()
