#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// ASCII characters sorted by brightness 
__constant__ char d_asciiTable[] = " .,:;o+*#%@"
#define ASCIISIZE sizeof(d_asciiTable)

extern "C" {

// Simplified Kernel: Maps a block of pixels to a single ASCII character (Monochrome)
// Removed color processing for optimization.
__global__ void videoToAsciiKernel(const uchar4* input, char* outputChar, int in_w, int in_h, int out_w, int out_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_w || y >= out_h) return;

    // Calculate which block of pixels this char corresponds to
    float scaleX = (float)in_w / out_w;
    float scaleY = (float)in_h / out_h;

    int startPixelsX = (int)(x * scaleX);
    int startPixelsY = (int)(y * scaleY);
    int endPixelsX = (int)((x + 1) * scaleX);
    int endPixelsY = (int)((y + 1) * scaleY);

    // Bounds check
    if (endPixelsX > in_w) endPixelsX = in_w;
    if (endPixelsY > in_h) endPixelsY = in_h;

    float totalBrightness = 0.0f;
    int count = 0;

    // Unroll loops slightly or keep simple. 
    // Optimization: Access global memory with coalescing is hard here due to block mapping.
    // However, given the downsampling, cache hits should be decent.
    for (int py = startPixelsY; py < endPixelsY; ++py) {
        for (int px = startPixelsX; px < endPixelsX; ++px) {
            uchar4 p = input[py * in_w + px];
            // Simple luminosity formula: 0.299R + 0.587G + 0.114B
            // Integer math is faster if we want to optimize further, but float is fine on GPU.
            float b = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
            totalBrightness += b;
            count++;
        }
    }

    if (count > 0) {
        totalBrightness /= count;
    }

    // Map Brightness to Char
    int index = (int)(totalBrightness / 255.0f * ASCIISIZE);
    if (index < 0) index = 0;
    if (index > ASCIISIZE) index = ASCIISIZE;
    outputChar[y * out_w + x] = d_asciiTable[index];
}

void launchVideoToAscii(const uchar4* d_input, char* d_outputChar, int in_w, int in_h, int out_w, int out_h) {
    dim3 threads(16, 16);
    dim3 blocks((out_w + threads.x - 1) / threads.x, (out_h + threads.y - 1) / threads.y);

    videoToAsciiKernel<<<blocks, threads>>>(d_input, d_outputChar, in_w, in_h, out_w, out_h);
}

}
