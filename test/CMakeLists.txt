include(../cmake/FindCatch2.cmake)

add_executable(tKernelCache tKernelCache.cc)
target_link_libraries(tKernelCache PRIVATE kernelwrappers::kernelwrappers)
target_link_libraries(tKernelCache PRIVATE Catch2::Catch2WithMain)
add_test(NAME tKernelCache COMMAND tKernelCache)
