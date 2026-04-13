#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kmeans.h"
#include <string.h>

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
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sqrt(sum);
}

double *kmeans(double *data, int num_points, int dim, int k, int max_iteration, int *clusters)
{
    // Allocate memory for centroids
    // centroids is a 1D array [k * dim]
    double *centroids = malloc(k * dim * sizeof(double));
    int *counts = malloc(k * sizeof(int));
    double *new_sums = malloc(k * dim * sizeof(double));
    memset(clusters, -1, num_points * sizeof(int));
    int centroid_changed = 0;

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

    #pragma omp parallel 
    {
        for (int iter = 0; iter < max_iteration; iter++)
        {
            #pragma omp single
            centroid_changed = 0;
            
            #pragma omp for reduction(| : centroid_changed)
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
            int stop = 0;
            #pragma omp single
            {
                if (!centroid_changed)
                {
                    printf("Converged at iteration %d\n", iter);
                    stop = 1;
                }
                memset(counts, 0, k * sizeof(int));
                memset(new_sums, 0, k * dim * sizeof(double));
            }

            #pragma omp barrier
            if (stop) {
                break;
            }
            
            // Calculate new centroids
            #pragma omp for schedule(static) reduction(+ : counts[ : k], new_sums[ : k * dim])
            for (int i = 0; i < num_points; i++)
            {
                int cid = clusters[i];
                for (int d = 0; d < dim; d++)
                {
                    new_sums[cid * dim + d] += data[i * dim + d];
                }
                counts[cid]++;
            }
            
            #pragma omp for schedule(static)
            for (int centroid = 0; centroid < k; centroid++)
            {
                if (counts[centroid] > 0)
                {
                    for (int d = 0; d < dim; d++)
                        centroids[centroid * dim + d] = new_sums[centroid * dim + d] / counts[centroid];
                }
            }
            
        }
    }   

    free(new_sums);
    free(counts);
    return centroids;

}