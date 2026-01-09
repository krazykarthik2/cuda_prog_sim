#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

extern "C" {

// Kernel to apply filters to video frame
__global__ void videoFilterKernel(const uchar4* input, uchar4* output, int w, int h, int filter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int idx = y * w + x;
    uchar4 p = input[idx];
    uchar4 out = p; // default original

    if (filter == 1) { // grayscale
        unsigned char gray = (unsigned char)(0.299f * p.x + 0.587f * p.y + 0.114f * p.z);
        out = make_uchar4(gray, gray, gray, p.w);
    } else if (filter == 2) { // inversion
        out = make_uchar4(255 - p.x, 255 - p.y, 255 - p.z, p.w);
    } else if (filter == 3) { // custom: simple sepia
        unsigned char r = (unsigned char)min(255.0f, 0.393f * p.x + 0.769f * p.y + 0.189f * p.z);
        unsigned char g = (unsigned char)min(255.0f, 0.349f * p.x + 0.686f * p.y + 0.168f * p.z);
        unsigned char b = (unsigned char)min(255.0f, 0.272f * p.x + 0.534f * p.y + 0.131f * p.z);
        out = make_uchar4(r, g, b, p.w);
    }

    output[idx] = out;
}

void launchVideoFilter(const uchar4* d_input, uchar4* d_output, int w, int h, int filter) {
    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    videoFilterKernel<<<blocks, threads>>>(d_input, d_output, w, h, filter);
}

}
