#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EMPTY 0
#define SAND 1
#define WALL 2

extern "C" {

__global__ void initGridKernel(int* grid, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w*h) {
        grid[idx] = EMPTY;
        // Walls around
        int x = idx % w;
        int y = idx / w;
        if (y == h - 1) grid[idx] = WALL;
        if (x == 0 || x == w - 1) grid[idx] = WALL;
        if (y == 0) grid[idx] = WALL; // Ceiling too?
    }
}

__device__ bool tryMove(int* grid, int srcIdx, int dstIdx) {
    // Attempt to swap 0 at dst with 1 (SAND)
    // expected: EMPTY (0). val: SAND (1).
    if (atomicCAS(&grid[dstIdx], EMPTY, SAND) == EMPTY) {
        // Success. We placed sand at dst. Now remove from src.
        grid[srcIdx] = EMPTY;
        return true;
    }
    return false;
}

__global__ void stepSandKernel(int* grid, int w, int h, int frame) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;

    if (grid[idx] == SAND) {
        int x = idx % w;
        int y = idx / w;

        if (y >= h - 1) return; // Bottom

        int down = idx + w;
        int downLeft = idx + w - 1;
        int downRight = idx + w + 1;

        // Try down
        if (tryMove(grid, idx, down)) return;

        // Try diagonals
        bool tryLeftFirst = (frame & 1) ^ (x & 1); // Randomize per frame and per column
        
        if (tryLeftFirst) {
            if (x > 0 && tryMove(grid, idx, downLeft)) return;
            if (x < w - 1 && tryMove(grid, idx, downRight)) return;
        } else {
            if (x < w - 1 && tryMove(grid, idx, downRight)) return;
            if (x > 0 && tryMove(grid, idx, downLeft)) return;
        }
    }
}

__global__ void renderGridKernel(int* grid, unsigned char* fb, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;

    int cell = grid[idx];
    int pixelIdx = idx * 4;

    unsigned char r=0, g=0, b=0;
    if (cell == SAND) {
        r=235; g=200; b=80;
        // Simple noise
        if ((idx ^ idx*13) % 5 == 0) { r-=10; g-=10; }
    } else if (cell == WALL) {
        r=128; g=128; b=128;
    } else {
        r=20; g=20; b=20;
    }

    fb[pixelIdx+0] = r;
    fb[pixelIdx+1] = g;
    fb[pixelIdx+2] = b;
    fb[pixelIdx+3] = 255;
}

__global__ void paintCircleKernel(int* grid, int w, int h, int cx, int cy, int r, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;
    int x = idx % w;
    int y = idx / w;
    int dx = x - cx;
    int dy = y - cy;
    if (dx*dx + dy*dy <= r*r) {
        if (grid[idx] != WALL) { // Don't overwrite walls unless we are painting walls?
            // Allow overwriting everything for now, or logic:
            grid[idx] = type;
        }
    }
}

__global__ void paintRectKernel(int* grid, int w, int h, int rx, int ry, int rw, int rh, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;
    int x = idx % w;
    int y = idx / w;
    if (x >= rx && x < rx + rw && y >= ry && y < ry + rh) {
        grid[idx] = type;
    }
}


// Host wrappers
void initGrid(int* grid, int w, int h) {
    initGridKernel<<<(w*h+255)/256, 256>>>(grid, w, h);
    cudaDeviceSynchronize();
}

void stepSand(int* grid, int w, int h, int frame) {
    stepSandKernel<<<(w*h+255)/256, 256>>>(grid, w, h, frame);
    // cudaDeviceSynchronize(); // logic synchronization not strictly needed between steps if efficient, but safety for now?
    // Actually, launch async is fine.
}

void renderSand(int* grid, unsigned char* fb, int w, int h) {
    renderGridKernel<<<(w*h+255)/256, 256>>>(grid, fb, w, h);
    // cudaDeviceSynchronize();
}

void paintCircle(int* grid, int w, int h, int cx, int cy, int r, int type) {
    paintCircleKernel<<<(w*h+255)/256, 256>>>(grid, w, h, cx, cy, r, type);
}

void paintRect(int* grid, int w, int h, int rx, int ry, int rw, int rh, int type) {
    paintRectKernel<<<(w*h+255)/256, 256>>>(grid, w, h, rx, ry, rw, rh, type);
}

}
