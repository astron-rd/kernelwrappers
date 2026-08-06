find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.3)
  FetchContent_MakeAvailable(Catch2)
endif()
