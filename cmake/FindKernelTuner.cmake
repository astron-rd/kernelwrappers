find_package(kernel_launcher QUIET)

if(NOT kernel_launcher_FOUND)
  FetchContent_Declare(
    kernel_launcher
    GIT_REPOSITORY https://github.com/KernelTuner/kernel_launcher.git
    GIT_TAG develop)
  FetchContent_MakeAvailable(kernel_launcher)

target_compile_options(kernel_launcher PRIVATE
    -Wno-error
  )

  # libkernelwrappers is a shared library, so the static kernel_launcher
  # archive must be position-independent code to link against it.
  set_target_properties(kernel_launcher PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(DEFINED kernel_launcher_SOURCE_DIR AND
   EXISTS "${kernel_launcher_SOURCE_DIR}/python/tune.py")
  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT)
    set(KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT
        "${kernel_launcher_SOURCE_DIR}/python/tune.py"
        CACHE FILEPATH "Kernel Launcher tuning entry point")
  endif()
  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY)
    set(KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY
        "${kernel_launcher_SOURCE_DIR}/python"
        CACHE PATH "Kernel Launcher Python package directory")
  endif()
else()
  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT)
    set(KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT
        ""
        CACHE FILEPATH "Kernel Launcher tuning entry point")
  endif()
  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY)
    set(KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY
        ""
        CACHE PATH "Kernel Launcher Python package directory")
  endif()
endif()
