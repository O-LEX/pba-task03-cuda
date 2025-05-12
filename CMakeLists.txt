cmake_minimum_required(VERSION 3.28)
project(tinycuda LANGUAGES CXX CUDA)

add_executable(tinycuda
    main.cu
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

target_include_directories(tinycuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

