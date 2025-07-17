#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <float.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/src/stb_image.h"
#include "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/src/common.h" //K clusters and N points defined
#include "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/src/timer.h"
#include "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/src/stb_image_write.h"
#define IMG_COUNT 10
#define CUFFT_CHECK(err) \
    do { \
        cufftResult err__ = (err); \
        if (err__ != CUFFT_SUCCESS) { \
            fprintf(stderr, "CUFFT Error at %s:%d - error code %d\n", __FILE__, __LINE__, err__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --CUDA Kernels --

//Sobel edge detection for contour edge detections and filtering
__global__ void sobel_filter_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;
    int gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
             + input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];
    int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
             + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
    int g = min(255, abs(gx) + abs(gy));
    output[y * width + x] = (unsigned char)g;
    if (x == 10 && y == 10) {
    printf("Sobel@(%d,%d): gx=%d, gy=%d, g=%d\n", x, y, gx, gy, g);
}
}
//Gaussian blur for noise reduction
__global__ void gaussian_blur_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    const float kernel[3][3] = {
        {1/16.f, 2/16.f, 1/16.f},
        {2/16.f, 4/16.f, 2/16.f},
        {1/16.f, 2/16.f, 1/16.f}
    };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int px = x + kx;
            int py = y + ky;
            sum += input[py * width + px] * kernel[ky + 1][kx + 1];
        }
    }

    output[y * width + x] = (unsigned char)sum;
}
//distance kernel function
__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

//clusters assigning (5 clusters are been defined)
__global__ void assign_clusters(Point *points, float *cx, float *cy, int numpoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numpoints) return;
    float min_dist = FLT_MAX;
    int cluster = -1;
    for (int k = 0; k < K; ++k) {
    float dx = points[i].x - cx[k];
    float dy = points[i].y - cy[k];
    float dist = dx * dx + dy * dy;
    if (dist < min_dist) {
        min_dist = dist;
        cluster = k;
    }
    }
    points[i].cluster = cluster;
    if (i < 5) { 
        printf("GPU assign_clusters: point[%d] = (%.1f, %.1f) → cluster %d\n",
               i, points[i].x, points[i].y, cluster);
    }
}

//centroid calculation based on assigned clusters distances
__global__ void compute_centroids(Point *points, float *cx, float *cy, int *counts, int numpoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numpoints) return;

    int loli = points[i].cluster;
    if (loli >= 0 && loli < K) {
        atomicAdd(&cx[loli], points[i].x);   // Sum of x for cluster c
        atomicAdd(&cy[loli], points[i].y);   // Sum of y for cluster c
        atomicAdd(&counts[loli], 1);         // Count of points in cluster c
    }
}


// --Image Generation--
int generate_points_from_image(Point *points, unsigned char *img, int w, int h) {
    float max_val = 0.0f, min_val = 255.0f;
    int histogram[256] = {0};
    for (int i = 0; i < w * h; ++i) {
        unsigned char val = img[i];
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
        histogram[val]++;
    }
    printf("Intensity range: min = %.2f, max = %.2f\n", min_val, max_val);
    float dynamic_threshold = 0.01f * max_val; 
    printf("Dynamic threshold set to: %.2f (1%% of max)\n", dynamic_threshold);
    int idx = 0;
    for (int y = 0; y < h && idx < N; ++y) {
        for (int x = 0; x < w && idx < N; ++x) {
            float intensity = img[y * w + x];
            if (intensity > dynamic_threshold) {
                points[idx].x = static_cast<float>(x);
                points[idx].y = static_cast<float>(y);
                points[idx].cluster = -1;
                idx++;
            }
        }
    }
    printf("Valid points found after thresholding: %d\n", idx);
    return idx;
}

int main() {
    GpuTimer timer;
    const char *imageFiles[IMG_COUNT] = {
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/baboon.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/coins.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/bird.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/buffalo.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/columns.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/dewey_defeats_truman.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/feep.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/ladyzhenskaya.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/snap.pgm",
    "/home/piyush/Downloads/Cuda-K_means/CUDAatScaleForTheEnterpriseCourseProjectTemplate/images/barbara.pgm"
};
    int w, h, c;
    unsigned char *h_img = nullptr;
    for(int i=0;i<IMG_COUNT;i++){
    h_img = stbi_load(imageFiles[i], &w, &h, &c, 1);
    if (!h_img) {
        fprintf(stderr, "Failed to load image: %s\n", imageFiles[i]);
        return -1;
    }
    printf("Loaded image: %s (%dx%d)\n", imageFiles[i], w, h);
    int fft_width = w;
    int fft_height = h;
    size_t real_size = w * h * sizeof(float);
    size_t complex_size = w * (h / 2 + 1) * sizeof(cufftComplex);
    float *d_img_real;
    cufftComplex *d_img_freq;
    float *d_img_ifft;
    CUDA_CHECK(cudaMalloc(&d_img_real, real_size));
    CUDA_CHECK(cudaMalloc(&d_img_freq, complex_size));
    CUDA_CHECK(cudaMalloc(&d_img_ifft, real_size));
    float *h_img_float = (float*)malloc(real_size);
    for (int i = 0; i < w * h; ++i)
        h_img_float[i] = (float)h_img[i];
    CUDA_CHECK(cudaMemcpy(d_img_real, h_img_float, real_size, cudaMemcpyHostToDevice));

    // FFT real-to-complex
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, fft_height, fft_width, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&plan_inv, fft_height, fft_width, CUFFT_C2R));

    // FFT Forward
    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_img_real, d_img_freq));

    // IFFT
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_img_freq, d_img_ifft));

    // Normalize copy image 
    float *h_ifft_out = (float *)malloc(real_size);
    unsigned char *h_img_ifft_u8 = (unsigned char *)malloc(w * h);
    CUDA_CHECK(cudaMemcpy(h_ifft_out, d_img_ifft, real_size, cudaMemcpyDeviceToHost));

    // Normalize
    float max_val = -FLT_MAX;
    float min_val = FLT_MAX;
    for (int i = 0; i < w * h; ++i) {
        h_ifft_out[i] /= (w * h);
        if (h_ifft_out[i] > max_val) max_val = h_ifft_out[i];
        if (h_ifft_out[i] < min_val) min_val = h_ifft_out[i];
    }
    for (int i = 0; i < w * h; ++i) {
        float val = (h_ifft_out[i] - min_val) / (max_val - min_val); // clamped to [0, 1]
        val *= 255.0f;
        h_img_ifft_u8[i] = (unsigned char)(fminf(fmaxf(val, 0.0f), 255.0f));
    }
    char ifft_filename[256];
    snprintf(ifft_filename, sizeof(ifft_filename), "ifft_image_%d.png", i);
    stbi_write_png(ifft_filename, w, h, 1, h_img_ifft_u8, w);
    printf("IFFT Saved: %s\n", ifft_filename);
    unsigned char *d_img_in, *d_img_tmp, *d_img_out;
    size_t img_size = w * h * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&d_img_in, img_size));
    CUDA_CHECK(cudaMalloc(&d_img_tmp, img_size));
    CUDA_CHECK(cudaMalloc(&d_img_out, img_size));
    CUDA_CHECK(cudaMemcpy(d_img_in, h_img_ifft_u8, img_size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    gaussian_blur_kernel<<<grid, block>>>(d_img_in, d_img_tmp, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    sobel_filter_kernel<<<grid, block>>>(d_img_tmp, d_img_out, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char *h_img_processed = (unsigned char*)malloc(img_size);
    CUDA_CHECK(cudaMemcpy(h_img_processed, d_img_out, img_size, cudaMemcpyDeviceToHost));
    
    char sobel_filename[256];
    snprintf(sobel_filename, sizeof(sobel_filename), "sobel_image_%d.png", i);
    stbi_write_png(sobel_filename, w, h, 1, h_img_processed, w);
    printf("Sobel Saved: %s\n", sobel_filename);

    //Points generation
    Point *h_points = (Point*)malloc(N * sizeof(Point));
    int valid_points = generate_points_from_image(h_points, h_img_processed, w, h);
    if (valid_points == 0) {
        printf("No valid points found after preprocessing. Try reducing intensity threshold.\n");
        return -1;
    }

    // K-Means processing
    Point *d_points;
    float *d_cx, *d_cy;
    int *d_counts;
    float h_cx[K], h_cy[K]; int h_counts[K];
    CUDA_CHECK(cudaMalloc(&d_points, valid_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_cx, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points, valid_points * sizeof(Point), cudaMemcpyHostToDevice));
    cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaMemcpy(d_cx, h_cx, K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, h_cy, K * sizeof(float), cudaMemcpyHostToDevice));
    for (int j = 0; j < K; ++j) {
        if (!isfinite(h_cx[j]) || !isfinite(h_cy[j])) {
            printf("Invalid centroid %d: (%.2f, %.2f)\n", j, h_cx[j], h_cy[j]);
        }
    }

    //Kernel CALL
    for (int i = 0; i < 50; ++i) {
    dim3 blockDim(256);
    dim3 gridDim((valid_points + blockDim.x - 1) / blockDim.x);
    timer.Start();
        assign_clusters<<<gridDim, blockDim>>>(d_points, d_cx, d_cy, valid_points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_cx, 0, K * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cy, 0, K * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(h_points, d_points, valid_points * sizeof(Point), cudaMemcpyDeviceToHost));

        compute_centroids<<<gridDim, blockDim>>>(d_points, d_cx, d_cy, d_counts, valid_points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cx, d_cx, K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cy, d_cy, K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost));
        srand(42);
        for (int j = 0; j < K; ++j) {
            if (h_counts[j] > 0) {
                h_cx[j] /= h_counts[j];
                h_cy[j] /= h_counts[j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_cx, h_cx, K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cy, h_cy, K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_points, d_points, valid_points * sizeof(Point), cudaMemcpyDeviceToHost));
        int countercl[K] = {0};
        for (int j = 0; j < valid_points; ++j) {
            int lol = h_points[j].cluster;
            if (lol >= 0 && lol < K) countercl[lol]++;
        }
        printf("Iteration %d:\n", i);
        for (int j = 0; j < K; ++j)
            printf("  Cluster %d has %d points\n", j, countercl[j]);
    }
    timer.Stop();
    printf("K-Means completed in %.3f ms.\n", timer.Elapsed());

    CUDA_CHECK(cudaMemcpy(h_points, d_points, valid_points * sizeof(Point), cudaMemcpyDeviceToHost));
    printf("Copied %d points from device\n", valid_points);
    for (int i = 0; i < min(20,valid_points); ++i) {
    if (h_points[i].cluster == -1) continue;
    }

    //clustered points
    unsigned char *cluster_rgb = (unsigned char *)calloc(w * h * 3, sizeof(unsigned char));
    unsigned char colors[8][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
        {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
        {128, 128, 0}, {255, 165, 0} 
    };
    for (int i = 0; i < valid_points; ++i) {
        int x = (int)h_points[i].x;
        int y = (int)h_points[i].y;
        int cluster = h_points[i].cluster;
        if (cluster >= 0 && cluster < K) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    int xx = x + dx;
                    int yy = y + dy;
                    if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                        int offset = (yy * w + xx) * 3;
                        cluster_rgb[offset + 0] = colors[cluster][0];
                        cluster_rgb[offset + 1] = colors[cluster][1];
                        cluster_rgb[offset + 2] = colors[cluster][2];
                    }
                }
            }
        }
    }
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "output_clusters_%d.png", i);
    int pixel_written = 0;
    for (int i = 0; i < w * h * 3; ++i) {
        if (cluster_rgb[i] != 0) pixel_written++;
    }
    printf("Pixels with nonzero color: %d\n", pixel_written);

    int success = stbi_write_png(output_filename, w, h, 3, cluster_rgb, w * 3);
    if (!success) {
        fprintf(stderr, "Failed to write PNG: %s\n", output_filename);
    }    
    else{
        printf("Clustered image saved as %s\n", output_filename);
    }
    // Cleanup
    CUDA_CHECK(cudaFree(d_img_in));
    CUDA_CHECK(cudaFree(d_img_tmp));
    CUDA_CHECK(cudaFree(d_img_out));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_cx));
    CUDA_CHECK(cudaFree(d_cy));
    CUDA_CHECK(cudaFree(d_counts));
    free(h_img);
    free(h_img_processed);
    free(h_points);
    CUDA_CHECK(cudaFree(d_img_real));
    CUDA_CHECK(cudaFree(d_img_freq));
    CUDA_CHECK(cudaFree(d_img_ifft));
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    free(h_ifft_out);
    free(h_img_ifft_u8);
    free(h_img_float);
    }
    return 0;
} 