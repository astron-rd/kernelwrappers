include_guard(GLOBAL)

include(CMakeParseArguments)
include(FetchContent)

# Generate a companion executable that captures kernels from TARGET and runs
# Kernel Launcher's tuner. TARGET must run a representative workload without
# command-line arguments.
function(kernelwrappers_add_tuner)
  cmake_parse_arguments(KWT "" "TARGET;OUTPUT_NAME;WISDOM_DIRECTORY" "" ${ARGN})

  if(NOT KWT_TARGET)
    message(FATAL_ERROR "kernelwrappers_add_tuner() requires TARGET")
  endif()
  if(NOT TARGET ${KWT_TARGET})
    message(FATAL_ERROR "kernelwrappers tuner target does not exist: ${KWT_TARGET}")
  endif()

  if(NOT KWT_OUTPUT_NAME)
    set(KWT_OUTPUT_NAME tune)
  endif()
  if(NOT KWT_WISDOM_DIRECTORY)
    set(KWT_WISDOM_DIRECTORY "${KERNELWRAPPERS_WISDOM_PATH}")
  endif()

  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT)
    message(FATAL_ERROR
            "Kernel Launcher's tune.py was not found; set KERNELWRAPPERS_KERNEL_LAUNCHER_TUNE_SCRIPT")
  endif()
  if(NOT DEFINED KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY)
    message(FATAL_ERROR "Kernel Launcher's Python package directory was not found")
  endif()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pycuda; import cupy"
    RESULT_VARIABLE python_packages_result
    OUTPUT_VARIABLE python_packages_output
    ERROR_VARIABLE python_packages_error)
  if(NOT python_packages_result EQUAL 0)
    message(FATAL_ERROR
            "kernelwrappers_add_tuner() requires Python packages pycuda and cupy. "
            "Interpreter: ${Python3_EXECUTABLE}\n${python_packages_error}")
  endif()

  if(NOT TARGET cxxopts::cxxopts)
    FetchContent_Declare(
      cxxopts
      GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
      GIT_TAG v3.2.1)
    FetchContent_MakeAvailable(cxxopts)
  endif()

  set(tuner_target "${KWT_TARGET}_tuner")
  if(TARGET ${tuner_target})
    message(FATAL_ERROR "a tuner has already been registered for ${KWT_TARGET}")
  endif()

  add_executable(${tuner_target} "${KERNELWRAPPERS_ROOT}/tools/Tune.cc")
  add_dependencies(${tuner_target} ${KWT_TARGET})
  target_link_libraries(${tuner_target} PRIVATE cxxopts::cxxopts)
  set_target_properties(
    ${tuner_target}
    PROPERTIES OUTPUT_NAME "${KWT_OUTPUT_NAME}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(
    ${tuner_target}
    PRIVATE KERNELWRAPPERS_TUNE_APPLICATION="$<TARGET_FILE:${KWT_TARGET}>"
            KERNELWRAPPERS_TUNE_PYTHON="${Python3_EXECUTABLE}"
            KERNELWRAPPERS_TUNE_SCRIPT="${KERNELWRAPPERS_ROOT}/tools/Tune.py"
            KERNELWRAPPERS_TUNE_KERNEL_LAUNCHER_PYTHON="${KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON_DIRECTORY}"
            KERNELWRAPPERS_TUNE_WISDOM_DIRECTORY="${KWT_WISDOM_DIRECTORY}")

  # A conventional build target makes the generated executable discoverable
  # through every CMake generator, including Makefiles and Ninja.
  if(NOT TARGET tune)
    add_custom_target(tune DEPENDS ${tuner_target})
  endif()
endfunction()
