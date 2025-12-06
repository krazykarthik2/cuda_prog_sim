#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EMPTY 0
#define WATER 1
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
        if (y == 0) grid[idx] = WALL;
    }
}

__device__ bool tryMove(int* grid, int srcIdx, int dstIdx) {
    if (atomicCAS(&grid[dstIdx], EMPTY, WATER) == EMPTY) {
        grid[srcIdx] = EMPTY;
        return true;
    }
    return false;
}

// 1. GRAVITY KERNEL: Strictly vertical movement
__global__ void updateGravityKernel(int* grid, int w, int h, int frame) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;

    // Only process Water
    if (grid[idx] == WATER) {
        int y = idx / w;
        if (y >= h - 1) return; // Bottom

        int down = idx + w;
        
        // Try falling straight down
        tryMove(grid, idx, down);
    }
}

// 2. SETTLE KERNEL: Diagonal and Horizontal movement
// Crucial: Only runs if the particle is supported from below to prevent mid-air jitter
__global__ void updateSettleKernel(int* grid, int w, int h, int frame) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;

    if (grid[idx] == WATER) {
        int x = idx % w;
        int y = idx / w;
        if (y >= h - 1) return;

        int down = idx + w;
        int cellBelow = grid[down];

        // LINEAR FALL FIX:
        // If the space below is EMPTY, we are falling. 
        // Do NOT move sideways or diagonal. Wait for Gravity kernel to handle us next frame.
        if (cellBelow == EMPTY) return;

        // If we are here, 'down' is blocked (WATER or WALL). We need to flow.
        
        int downLeft = idx + w - 1;
        int downRight = idx + w + 1;
        int left = idx - 1;
        int right = idx + 1;

        // Pseudo-random direction to avoid bias
        bool tryLeftFirst = (frame & 1) ^ (x & 1) ^ (y & 1); 

        // 1. Try Diagonals (Flowing down slopes)
        if (tryLeftFirst) {
            if (x > 0 && tryMove(grid, idx, downLeft)) return;
            if (x < w - 1 && tryMove(grid, idx, downRight)) return;
        } else {
            if (x < w - 1 && tryMove(grid, idx, downRight)) return;
            if (x > 0 && tryMove(grid, idx, downLeft)) return;
        }

        // 2. Try Horizontal (Spreading / Leveling)
        if (tryLeftFirst) {
             if (x > 0 && tryMove(grid, idx, left)) return;
             if (x < w - 1 && tryMove(grid, idx, right)) return;
        } else {
             if (x < w - 1 && tryMove(grid, idx, right)) return;
             if (x > 0 && tryMove(grid, idx, left)) return;
        }
    }
}

__global__ void renderGridKernel(int* grid, unsigned char* fb, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;

    int cell = grid[idx];
    int pixelIdx = idx * 4;

    unsigned char r=0, g=0, b=0;
    if (cell == WATER) {
        // More realistic water color (Deep Blue)
        r=20; g=100; b=220;
        
        // Add some shimmer based on index and depth (optional visual polish)
        if ((idx * 17 + blockIdx.x) % 31 < 5) {
             r += 20; g += 30; b += 30;
        }
    } else if (cell == WALL) {
        r=100; g=100; b=100;
    } else {
        r=10; g=10; b=10;
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
        if (grid[idx] != WALL) {
            // Stability check: 
            // If painting WATER, only paint on EMPTY.
            // If painting WALL/EMPTY (Eraser), overwrite anything.
            if (type == WATER) {
                if (grid[idx] == EMPTY) grid[idx] = WATER;
            } else {
                grid[idx] = type;
            }
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

void stepFluids(int* grid, int w, int h, int frame, int gravitySteps, int flowSteps) {
    dim3 gridDim((w*h+255)/256);
    dim3 blockDim(256);

    // To ensure "smoother" simulation, we can alternate small steps
    // But to satisfy "gravity slider", we run gravity kernel N times
    // And "flow slider", we run settle kernel N times.
    
    // Strict separation ensures falling particles don't disperse
    
    for(int g=0; g<gravitySteps; ++g) {
        updateGravityKernel<<<gridDim, blockDim>>>(grid, w, h, frame + g);
        cudaDeviceSynchronize();
    }

    for(int f=0; f<flowSteps; ++f) {
        updateSettleKernel<<<gridDim, blockDim>>>(grid, w, h, frame + f);
        cudaDeviceSynchronize();
    }
}

void renderFluids(int* grid, unsigned char* fb, int w, int h) {
    renderGridKernel<<<(w*h+255)/256, 256>>>(grid, fb, w, h);
}

void paintCircle(int* grid, int w, int h, int cx, int cy, int r, int type) {
    paintCircleKernel<<<(w*h+255)/256, 256>>>(grid, w, h, cx, cy, r, type);
}

void paintRect(int* grid, int w, int h, int rx, int ry, int rw, int rh, int type) {
    paintRectKernel<<<(w*h+255)/256, 256>>>(grid, w, h, rx, ry, rw, rh, type);
}

}
