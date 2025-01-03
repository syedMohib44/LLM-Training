cmake_minimum_required(VERSION 3.10)
project(CNNWithGPU)

# Ensure C++17
set(CMAKE_CXX_STANDARD 17)

# Set the paths to libtorch and CUDA
set(Torch_DIR "C:/libtorch/share/cmake/Torch")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")

# Find Torch and CUDA
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Add your executable and link libraries
add_executable(neural_network_cuda neural_network_cuda.cpp)
target_link_libraries(neural_network_cuda "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

# Enable GPU
if (TORCH_CUDA_AVAILABLE)
    target_compile_definitions(neural_network_cuda PUBLIC TORCH_CUDA_AVAILABLE)
endif()

# Ensure the appropriate C++ standard is used
set_property(TARGET neural_network_cuda PROPERTY CXX_STANDARD 17)

# Post-build step to copy libraries
add_custom_command(TARGET neural_network_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/libtorch/lib" "$<TARGET_FILE_DIR:neural_network_cuda>/lib"
    COMMENT "Copying required libraries to the output directory"
)