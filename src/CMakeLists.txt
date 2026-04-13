add_library(optima_kmeans optima_kmeans.c dataloader.c kmeans.c gpu_implementation/kmeans_gpu.cu)
target_include_directories(optima_kmeans PUBLIC ${CMAKE_SOURCE_DIR}/include)
