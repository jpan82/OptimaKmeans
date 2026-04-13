#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kmeans.h"
#include <string.h>
#include <omp.h>

/**
 * @brief Calculates the Euclidean distance between two points
 *
 * @param p1 Point one's feature vector
 * @param p2 Point two's feature vector
 * @param dim Number of dimensions (features) per point
 * @return double The Euclidean distance between the two points
 */
static double dist(double *p1, double *p2, int dim)
{
    double sum = 0;
    // Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + ...)
    for (int i = 0; i < dim; i++)
    {
        sum += (p1[i] - p2[i])*(p1[i] - p2[i]);
    }
    return sqrt(sum);
}

double *kmeans(double *data, int num_points, int dim, int k, int max_iteration, int *clusters)
{
    // Allocate memory for centroids
    // centroids is a 1D array [k * dim]
    double *centroids = malloc(k * dim * sizeof(double));

    memset(clusters, -1, num_points * sizeof(int));

    // Pick random points as starting centroids
    // srand(time(NULL));
    for (int i = 0; i < k; i++)
    {
        int r = rand() % num_points;
        for (int d = 0; d < dim; d++)
        {
            centroids[i * dim + d] = data[r * dim + d];
        }
    }

    for (int iter = 0; iter < max_iteration; iter++)
    {
        int centroid_changed = 0;

        // Find closest centroid for each point
        #pragma omp parallel for reduction(|:centroid_changed)
        for (int i = 0; i < num_points; i++)
        {
            double min_d = 1e18;
            // The index of the closest centroid for each point
            int closest_centroid = 0;
            // Calculate distance of this point to each centroid and find the closest one
            for (int centroid = 0; centroid < k; centroid++)
            {
                double distance = dist(&data[i * dim], &centroids[centroid * dim], dim);
                if (distance < min_d)
                {
                    min_d = distance;
                    closest_centroid = centroid;
                }
            }
            // If the closest centroid is different from the current cluster assignment, update it
            if (clusters[i] != closest_centroid)
            {
                clusters[i] = closest_centroid;
                centroid_changed = 1;
            }
        }

        // If no points changed clusters, we have converged
        //第一版
        /*
        int *counts = calloc(k, sizeof(int));
        double *new_sums = calloc(k * dim, sizeof(double));

        int nthreads = omp_get_max_threads();
        double *all_local_sums = calloc(nthreads * k * dim, sizeof(double));
        int *all_local_counts = calloc(nthreads * k, sizeof(int));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *local_sums = &all_local_sums[tid * k * dim];
            int *local_counts = &all_local_counts[tid * k];

            #pragma omp for schedule(static)
            for (int i = 0; i < num_points; i++)
            {
                int cid = clusters[i];
                for (int d = 0; d < dim; d++) {
                    local_sums[cid * dim + d] += data[i * dim + d];
                }
                local_counts[cid]++;
            }
        }

        for (int t = 0; t < nthreads; t++) {
            for (int c = 0; c < k; c++) {
                counts[c] += all_local_counts[t * k + c];
                for (int d = 0; d < dim; d++) {
                    new_sums[c * dim + d] += all_local_sums[t * k * dim + c * dim + d];
                }
            }
        }

        free(all_local_sums);
        free(all_local_counts);
        */
        //第一版结束


        //第二版atom
        /*
        int *counts = calloc(k, sizeof(int));
        double *new_sums = calloc(k * dim, sizeof(double));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_points; i++)
        {
            int cid = clusters[i];

            for (int d = 0; d < dim; d++) {
                #pragma omp atomic
                new_sums[cid * dim + d] += data[i * dim + d];
            }

            #pragma omp atomic
            counts[cid]++;
        }
        */
        //第二版结束


        //第三版全reduction
        
        int *counts = calloc(k, sizeof(int));
        double *new_sums = calloc(k * dim, sizeof(double));

        #pragma omp parallel for schedule(static) reduction(+:counts[:k], new_sums[:k*dim])
        for (int i = 0; i < num_points; i++)
        {
            int cid = clusters[i];
            for (int d = 0; d < dim; d++) {
                new_sums[cid * dim + d] += data[i * dim + d];
            }
            counts[cid]++;
        }


        // Update centroids by calculating the mean of the points assigned to each cluster
        #pragma omp parallel for schedule(static)
        for (int centroid = 0; centroid < k; centroid++)
        {
            if (counts[centroid] > 0)
            {
                for (int d = 0; d < dim; d++)
                    centroids[centroid * dim + d] = new_sums[centroid * dim + d] / counts[centroid];
            }
        }
        free(new_sums);
        free(counts);
    }
    return centroids;
}