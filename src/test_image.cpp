#include <iostream>
#include <cuda_runtime.h>
#include <nppi.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../src/stb_image.h"

int main() {
    const char* imagePath = "../images/baboon.pgm";
    int width, height, channels;

    // Load grayscale image
    unsigned char* h_img = stbi_load(imagePath, &width, &height, &channels, 1);
    if (!h_img) {
        std::cerr << "❌ Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    std::cout << "✅ Loaded image: " << imagePath << " (" << width << "x" << height << ")\n";

    int imageSize = width * height;

    // Allocate device memory
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;

    cudaMalloc((void**)&d_src, imageSize);
    cudaMalloc((void**)&d_dst, imageSize);

    cudaMemcpy(d_src, h_img, imageSize, cudaMemcpyHostToDevice);

    // Setup size
    NppiSize roiSize = { width, height };
    int srcStep = width;
    int dstStep = width;

    // Run NPP copy
    NppStatus status = nppiCopy_8u_C1R(d_src, srcStep, d_dst, dstStep, roiSize);

    if (status == NPP_SUCCESS) {
        std::cout << "✅ NPP Copy succeeded.\n";
    } else {
        std::cerr << "❌ NPP Copy failed with status: " << status << "\n";
    }

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(h_img);

    std::cout << "Test completed.\n";
    return 0;
}
