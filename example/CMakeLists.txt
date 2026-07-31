add_executable(kernelwrappers_example main.cc ExampleProcessingStep.cc
                                      ExampleKernel.cc)
target_link_libraries(kernelwrappers_example
                      PRIVATE kernelwrappers::kernelwrappers)

# Embed the CUDA kernel source into the binary (provided by cudawrappers). This
# makes the source available as the std::string ExampleProcessingStep_source via
# the generated header example/kernels/ExampleProcessingStep.cu.o.h.
target_embed_source(kernelwrappers_example "kernels/ExampleProcessingStep.cu")
