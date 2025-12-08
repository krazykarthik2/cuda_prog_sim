#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EMPTY 0
#define WATER 1
#define WALL  2

extern "C" {

// =====================================================
// INIT GRID
// =====================================================

__global__ void initGridKernel(int* grid, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    int x = idx % w;
    int y = idx / w;

    grid[idx] = EMPTY;

    if (y == 0 || y == h - 1 || x == 0 || x == w - 1)
        grid[idx] = WALL;
}

__device__ __forceinline__ bool tryMove(int* grid, int src, int dst) {
    if (atomicCAS(&grid[dst], EMPTY, WATER) == EMPTY) {
        grid[src] = EMPTY;
        return true;
    }
    return false;
}

// =====================================================
// 1. GRAVITY KERNEL (STRICT FREE FALL)
// =====================================================

__global__ void updateGravityKernel(int* grid, int w, int h, int frame) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    if (grid[idx] != WATER) return;

    int y = idx / w;
    if (y >= h - 1) return;

    int down = idx + w;
    tryMove(grid, idx, down);
}

// =====================================================
// 2. SETTLE / FLOW KERNEL (OPTIMIZED)
// =====================================================

__global__ void updateSettleKernel(int* grid, int w, int h, int frame) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;
    if (grid[idx] != WATER) return;

    int x = idx % w;
    int y = idx / w;
    if (y >= h - 1) return;

    int down = idx + w;
    int below = grid[down];
    int belowBelow = (y + 2 < h) ? grid[idx + 2 * w] : WALL;

    // ✅ TRUE FREE-FALL DETECTION (O(1))
    bool stableSupport = (below == WALL) || (below == WATER && belowBelow != EMPTY);
    if (!stableSupport) return;

    int downLeft  = idx + w - 1;
    int downRight = idx + w + 1;

    bool tryLeftFirst = (frame ^ x ^ y) & 1;

    // ✅ DIAGONAL FLOW (SLOPE BEHAVIOR)
    if (tryLeftFirst) {
        if (x > 0     && tryMove(grid, idx, downLeft))  return;
        if (x < w - 1 && tryMove(grid, idx, downRight)) return;
    } else {
        if (x < w - 1 && tryMove(grid, idx, downRight)) return;
        if (x > 0     && tryMove(grid, idx, downLeft))  return;
    }

    // ✅ FAST PRESSURE-BASED HORIZONTAL FLOW
    int left  = idx - 1;
    int right = idx + 1;

    bool leftEmpty  = (x > 0)   && grid[left]  == EMPTY;
    bool rightEmpty = (x < w-1) && grid[right] == EMPTY;

    if (tryLeftFirst) {
        if (leftEmpty  && tryMove(grid, idx, left))  return;
        if (rightEmpty && tryMove(grid, idx, right)) return;
    } else {
        if (rightEmpty && tryMove(grid, idx, right)) return;
        if (leftEmpty  && tryMove(grid, idx, left))  return;
    }
}

// =====================================================
// RENDER KERNEL
// =====================================================

__global__ void renderGridKernel(int* grid, unsigned char* fb, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    int cell = grid[idx];
    int p = idx * 4;

    unsigned char r = 10, g = 10, b = 10;

    if (cell == WATER) {
        r = 20; g = 50; b = 220;
    }
    else if (cell == WALL) {
        r = 100; g = 100; b = 100;
    }

    fb[p + 0] = r;
    fb[p + 1] = g;
    fb[p + 2] = b;
    fb[p + 3] = 255;
}

// =====================================================
// PAINT KERNELS
// =====================================================

__global__ void paintCircleKernel(int* grid, int w, int h,
                                  int cx, int cy, int r, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    int x = idx % w;
    int y = idx / w;

    int dx = x - cx;
    int dy = y - cy;

    if (dx * dx + dy * dy <= r * r) {
        if (grid[idx] != WALL) {
            if (type == WATER) {
                if (grid[idx] == EMPTY)
                    grid[idx] = WATER;
            } else {
                grid[idx] = type;
            }
        }
    }
}

__global__ void paintRectKernel(int* grid, int w, int h,
                                int rx, int ry, int rw, int rh, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    int x = idx % w;
    int y = idx / w;

    if (x >= rx && x < rx + rw && y >= ry && y < ry + rh)
        grid[idx] = type;
}

// =====================================================
// HOST WRAPPERS (OPTIMIZED SYNC MODEL)
// =====================================================

void initGrid(int* grid, int w, int h) {
    dim3 blocks((w * h + 255) / 256);
    dim3 threads(256);
    initGridKernel<<<blocks, threads>>>(grid, w, h);
    cudaDeviceSynchronize();
}

void stepFluids(int* grid, int w, int h,
                int frame, int gravitySteps, int flowSteps) {

    dim3 blocks((w * h + 255) / 256);
    dim3 threads(256);

    // ✅ NO PER-ITERATION SYNCS
    for (int g = 0; g < gravitySteps; ++g)
        updateGravityKernel<<<blocks, threads>>>(grid, w, h, frame + g);

    for (int f = 0; f < flowSteps; ++f)
        updateSettleKernel<<<blocks, threads>>>(grid, w, h, frame + f);

    // ✅ ONE SYNC PER FRAME
    cudaDeviceSynchronize();
}

void renderFluids(int* grid, unsigned char* fb, int w, int h) {
    dim3 blocks((w * h + 255) / 256);
    dim3 threads(256);
    renderGridKernel<<<blocks, threads>>>(grid, fb, w, h);
}

void paintCircle(int* grid, int w, int h,
                 int cx, int cy, int r, int type) {
    dim3 blocks((w * h + 255) / 256);
    dim3 threads(256);
    paintCircleKernel<<<blocks, threads>>>(grid, w, h, cx, cy, r, type);
}

void paintRect(int* grid, int w, int h,
               int rx, int ry, int rw, int rh, int type) {
    dim3 blocks((w * h + 255) / 256);
    dim3 threads(256);
    paintRectKernel<<<blocks, threads>>>(grid, w, h, rx, ry, rw, rh, type);
}

} // extern "C"
