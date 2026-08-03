find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.13.0)
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(tKernelCache tKernelCache.cc)
target_link_libraries(tKernelCache PRIVATE kernelwrappers::kernelwrappers)
target_link_libraries(tKernelCache PRIVATE Catch2::Catch2WithMain)
add_test(NAME tKernelCache COMMAND tKernelCache)
