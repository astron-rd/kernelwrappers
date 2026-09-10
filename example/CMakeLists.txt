add_executable(kernelwrappers_example main.cc ExampleProcessingStep.cc
                                      ScaleKernel.cc OffsetKernel.cc)
target_link_libraries(kernelwrappers_example
                      PRIVATE kernelwrappers::kernelwrappers)

target_embed_source(kernelwrappers_example "kernels/ScaleKernel.cu")
target_embed_source(kernelwrappers_example "kernels/OffsetKernel.cu")

if(KERNELWRAPPERS_TUNER)
  kernelwrappers_add_tuner(TARGET kernelwrappers_example)
endif()
