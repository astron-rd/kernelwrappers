find_package(kernel_launcher QUIET)

if(NOT kernel_launcher_FOUND)
  FetchContent_Declare(
    kernel_launcher
    GIT_REPOSITORY https://github.com/KernelTuner/kernel_launcher.git
    GIT_TAG develop)
  FetchContent_MakeAvailable(kernel_launcher)

  target_compile_options(kernel_launcher PRIVATE -Wno-error)

  # libkernelwrappers is a shared library, so the static kernel_launcher archive
  # must be position-independent code to link against it.
  set_target_properties(kernel_launcher PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT DEFINED kernel_launcher_SOURCE_DIR
   OR NOT EXISTS "${kernel_launcher_SOURCE_DIR}/python")
  message(FATAL_ERROR "Kernel Launcher Python package directory was not found")
endif()

set(KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY
    "${kernel_launcher_SOURCE_DIR}/python")
