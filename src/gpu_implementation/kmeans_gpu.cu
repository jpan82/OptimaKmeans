#include "kmeans_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, int N, int D, int K)
{
    //TODO
}

// Calculating the centroids
__global__ void centroid_sum(double *d_data, int *d_clusters, double *d_new_centroids, int *d_counts, int N, int D, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int cid = d_clusters[idx];
    if (cid < 0 || cid >= K) return;
    unsigned int active = __activemask();
    unsigned group = __match_any_sync(active, cid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1;
    for (int d = 0; d < D; ++d) {
        double v = d_data[idx * D + d];
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(active, v, offset);
            int partner = lane + offset;
            if (partner < 32 && ((group >> partner) & 1))
                v += other;
        }
        if (lane==leader) atomicAdd(&d_new_centroids[cid * D + d], v);
    }
    if (lane==leader) atomicAdd(&d_counts[cid], __popc(group));
}

__global__ void calculate_centroid(double *d_new_centroids, int *d_counts, int D, int K)
{
    int cid = blockIdx.x;
    int d = threadIdx.x;
    if (cid >= K || d >= D) return;
    int count = d_counts[cid];
    if (count > 0) {
        d_new_centroids[cid * D + d] /= count;
    }
    
}

double* kmeans_gpu(double *d_data, int num_points, int dim, int k, int max_iteration, int *d_clusters)
{
    double *d_new_centroids;
    double *d_counts;
    cudaMalloc(&d_new_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_counts, k * sizeof(int));
    cudaMemset(d_new_centroids, 0, k * dim * sizeof(double));
    cudaMemset(d_counts, 0, k * sizeof(int));
    find_centroid<<<(num_points + 255) / 256, 256>>>(d_data, d_new_centroids, d_clusters, num_points, dim, k);
    centroid_sum<<<(num_points + 255) / 256, 256>>>(d_data, d_clusters, d_new_centroids, d_counts, num_points, dim, k);
    calculate_centroid<<<k,dim>>>(d_new_centroids,d_counts,k,dim);
    return d_new_centroids;
}