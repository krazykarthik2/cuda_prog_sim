#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

extern "C" {

// Convolution kernels
__constant__ float gaussian3[9] = {1.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 4.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 1.0f/16};
__constant__ float gaussian5[25] = {
    1.0f/256, 4.0f/256, 6.0f/256, 4.0f/256, 1.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    1.0f/256, 4.0f/256, 6.0f/256, 4.0f/256, 1.0f/256
};
__constant__ float laplacian[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
__constant__ float mean3[9] = {1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9};
__constant__ float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__constant__ float sharpen[9] = {0, -2, 0, -2, 9, -2, 0, -2, 0};

// Kernel to apply filters to video frame
__global__ void videoFilterKernel(const uchar4* input, uchar4* output, int w, int h, int filter, int radius) {
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
    } else if (filter >= 4 && filter <= 8) {
        // Convolution filters
        const int HALO = 2;
        const int BLOCK_SIZE = 16;
        __shared__ float sharedR[BLOCK_SIZE + 2*HALO][BLOCK_SIZE + 2*HALO];
        __shared__ float sharedG[BLOCK_SIZE + 2*HALO][BLOCK_SIZE + 2*HALO];
        __shared__ float sharedB[BLOCK_SIZE + 2*HALO][BLOCK_SIZE + 2*HALO];

        int globalX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int globalY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

        int sidx_y = threadIdx.y + HALO;
        int sidx_x = threadIdx.x + HALO;

        int globalX_start = blockIdx.x * BLOCK_SIZE - HALO;
        int globalY_start = blockIdx.y * BLOCK_SIZE - HALO;
        int tileSize = BLOCK_SIZE + 2 * HALO;

        for (int i = threadIdx.x; i < tileSize; i += BLOCK_SIZE) {
            for (int j = threadIdx.y; j < tileSize; j += BLOCK_SIZE) {
                int globalX = globalX_start + i;
                int globalY = globalY_start + j;
                if (globalX >= 0 && globalX < w && globalY >= 0 && globalY < h) {
                    int idx = globalY * w + globalX;
                    sharedR[j][i] = input[idx].x;
                    sharedG[j][i] = input[idx].y;
                    sharedB[j][i] = input[idx].z;
                } else {
                    sharedR[j][i] = 0;
                    sharedG[j][i] = 0;
                    sharedB[j][i] = 0;
                }
            }
        }

        __syncthreads();

        float sumR = 0, sumG = 0, sumB = 0;
        const float* kern = nullptr;
        int ksize = 0;

        if (filter == 4) { // Gaussian blur
            if (radius <= 10) {
                kern = gaussian3;
                ksize = 3;
            } else {
                kern = gaussian5;
                ksize = 5;
            }
        } else if (filter == 5) { // Sharpen
            kern = sharpen;
            ksize = 3;
        } else if (filter == 6) { // High-pass
            kern = laplacian;
            ksize = 3;
        } else if (filter == 7) { // Low-pass
            kern = mean3;
            ksize = 3;
        } else if (filter == 8) { // Edge detection
            // Compute Gx and Gy on gray
            float gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int kidx = (ky + 1) * 3 + (kx + 1);
                    float g = 0.299f * sharedR[sidx_y + ky][sidx_x + kx] + 0.587f * sharedG[sidx_y + ky][sidx_x + kx] + 0.114f * sharedB[sidx_y + ky][sidx_x + kx];
                    gx += g * sobelX[kidx];
                    gy += g * sobelY[kidx];
                }
            }
            float mag = sqrtf(gx * gx + gy * gy);
            mag = fminf(mag / 4.0f, 255.0f);
            out = make_uchar4(mag, mag, mag, p.w);
        }

        if (kern) {
            int offset = ksize / 2;
            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int kidx = (ky + offset) * ksize + (kx + offset);
                    sumR += sharedR[sidx_y + ky][sidx_x + kx] * kern[kidx];
                    sumG += sharedG[sidx_y + ky][sidx_x + kx] * kern[kidx];
                    sumB += sharedB[sidx_y + ky][sidx_x + kx] * kern[kidx];
                }
            }
            if (filter == 5) { // Sharpen with strength
                float strength = radius / 10.0f;
                sumR = p.x + (sumR - p.x) * strength;
                sumG = p.y + (sumG - p.y) * strength;
                sumB = p.z + (sumB - p.z) * strength;
            }
            out = make_uchar4(fminf(fmaxf(sumR, 0.0f), 255.0f), fminf(fmaxf(sumG, 0.0f), 255.0f), fminf(fmaxf(sumB, 0.0f), 255.0f), p.w);
        }
    }

    output[idx] = out;
}

void launchVideoFilter(const uchar4* d_input, uchar4* d_output, int w, int h, int filter, int radius) {
    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    videoFilterKernel<<<blocks, threads>>>(d_input, d_output, w, h, filter, radius);
}

}
