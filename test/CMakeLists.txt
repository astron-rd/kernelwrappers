include(../cmake/FindCatch2.cmake)

add_executable(tKernelCache tKernelCache.cc)
target_link_libraries(tKernelCache PRIVATE kernelwrappers::kernelwrappers)
target_link_libraries(tKernelCache PRIVATE Catch2::Catch2WithMain)
add_test(NAME tKernelCache COMMAND tKernelCache)

if(KERNELWRAPPERS_TUNER)
  add_executable(tOnlineTuning tOnlineTuning.cc)
  target_link_libraries(tOnlineTuning PRIVATE kernelwrappers::kernelwrappers)
  target_link_libraries(tOnlineTuning PRIVATE Catch2::Catch2WithMain)
  add_test(NAME tOnlineTuning COMMAND tOnlineTuning)

  add_executable(tFailedOnlineTuning tFailedOnlineTuning.cc)
  target_link_libraries(tFailedOnlineTuning
                        PRIVATE kernelwrappers::kernelwrappers)
  target_link_libraries(tFailedOnlineTuning PRIVATE Catch2::Catch2WithMain)
  add_test(NAME tFailedOnlineTuning COMMAND tFailedOnlineTuning)
endif()
