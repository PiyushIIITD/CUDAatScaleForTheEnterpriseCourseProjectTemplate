// test_npp_basic.cu
#include <iostream>
#include <nppi.h>
#include <cuda_runtime.h>

int main() {
    const int width = 32, height = 32;
    const int pitch = width;
    const NppiSize size = { width, height };

    Npp8u *d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(Npp8u));

    // Fill image with 255 using NPP
    NppStatus status = nppiSet_8u_C1R(255, d_image, pitch, size);

    if (status == NPP_SUCCESS) {
        std::cout << "✅ NPP basic function (nppiSet) works correctly.\n";
    } else {
        std::cerr << "❌ NPP basic function failed. Status: " << status << "\n";
    }

    cudaFree(d_image);
    return 0;
}
