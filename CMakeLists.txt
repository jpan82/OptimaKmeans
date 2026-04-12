cmake_minimum_required(VERSION 3.10)
project(OptimaKmeans C)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_executable(example main.c)
target_link_libraries(example optima_kmeans m)