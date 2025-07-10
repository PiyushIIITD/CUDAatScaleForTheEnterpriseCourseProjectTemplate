// File: cuda_kmeans.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <npp.h>  // NPP image processing
#include <nppcore.h>
#include "src/common.h"  // Common helper definitions
#include "src/timer.h"   // Timing utilities

#define N 10000     // Number of points
#define K 5         // Number of clusters
#define MAX_ITERS 100

struct Point {
    float x, y;
    int cluster;
};

__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

__global__ void assign_clusters(Point *points, float *centroids_x, float *centroids_y, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_dist = 1e10;
    int cluster = 0;
    for (int j = 0; j < K; ++j) {
        float d = distance(points[i].x, points[i].y, centroids_x[j], centroids_y[j]);
        if (d < min_dist) {
            min_dist = d;
            cluster = j;
        }
    }
    points[i].cluster = cluster;
}

__global__ void compute_centroids(Point *points, float *centroids_x, float *centroids_y, int *counts, int N, int K) {
    __shared__ float temp_x[K];
    __shared__ float temp_y[K];
    __shared__ int temp_count[K];

    int tid = threadIdx.x;

    if (tid < K) {
        temp_x[tid] = 0.0f;
        temp_y[tid] = 0.0f;
        temp_count[tid] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int cluster = points[i].cluster;
        atomicAdd(&temp_x[cluster], points[i].x);
        atomicAdd(&temp_y[cluster], points[i].y);
        atomicAdd(&temp_count[cluster], 1);
    }
    __syncthreads();

    if (tid < K && temp_count[tid] > 0) {
        centroids_x[tid] = temp_x[tid] / temp_count[tid];
        centroids_y[tid] = temp_y[tid] / temp_count[tid];
        counts[tid] = temp_count[tid];
    }
}

void generatePoints(Point *points) {
    for (int i = 0; i < N; ++i) {
        points[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        points[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        points[i].cluster = -1;
    }
}

void applyNPPImageOps() {
    const int width = 512, height = 512;
    Npp8u *src = nppiMalloc_8u_C1(width, height, nullptr);
    Npp8u *dst = nppiMalloc_8u_C1(width, height, nullptr);

    NppStatus status = nppiThreshold_Val_8u_C1R(src, width, dst, width, {width, height}, 128);
    if (status != NPP_SUCCESS) {
        printf("NPP threshold operation failed.\n");
    } else {
        printf("NPP threshold operation succeeded.\n");
    }

    nppiFree(src);
    nppiFree(dst);
}

int main() {
    GpuTimer timer;

    Point *h_points = (Point*)malloc(N * sizeof(Point));
    generatePoints(h_points);

    Point *d_points;
    float *d_centroids_x, *d_centroids_y;
    int *d_counts;

    cudaMalloc(&d_points, N * sizeof(Point));
    cudaMalloc(&d_centroids_x, K * sizeof(float));
    cudaMalloc(&d_centroids_y, K * sizeof(float));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_points, h_points, N * sizeof(Point), cudaMemcpyHostToDevice);

    float h_centroids_x[K], h_centroids_y[K];
    for (int i = 0; i < K; ++i) {
        h_centroids_x[i] = h_points[i].x;
        h_centroids_y[i] = h_points[i].y;
    }

    cudaMemcpy(d_centroids_x, h_centroids_x, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_y, h_centroids_y, K * sizeof(float), cudaMemcpyHostToDevice);

    timer.Start();
    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        assign_clusters<<<(N+255)/256, 256>>>(d_points, d_centroids_x, d_centroids_y, N, K);
        cudaDeviceSynchronize();

        compute_centroids<<<(N+255)/256, 256>>>(d_points, d_centroids_x, d_centroids_y, d_counts, N, K);
        cudaDeviceSynchronize();
    }
    timer.Stop();

    printf("K-Means completed in %.3f ms.\n", timer.Elapsed());

    cudaMemcpy(h_points, d_points, N * sizeof(Point), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("Point (%.2f, %.2f) -> Cluster %d\n", h_points[i].x, h_points[i].y, h_points[i].cluster);
    }

    applyNPPImageOps();

    cudaFree(d_points);
    cudaFree(d_centroids_x);
    cudaFree(d_centroids_y);
    cudaFree(d_counts);
    free(h_points);

    return 0;
}
