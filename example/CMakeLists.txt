add_executable(kernelwrappers_example main.cc ExampleProcessingStep.cc
                                      ExampleKernel.cc)
target_link_libraries(kernelwrappers_example
                      PRIVATE kernelwrappers::kernelwrappers)

target_embed_source(kernelwrappers_example "kernels/ExampleProcessingStep.cu")
