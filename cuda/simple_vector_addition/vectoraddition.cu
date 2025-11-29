// /c:/Users/Allagadda radesh/kan_expirement_karthik/cuda/vectoraddition.cu
//
// Simple, memory-optimized matrix addition with synchronization and timing.
// - Uses pinned host memory for faster transfers
// - Uses 2D grid/block with strided loops for arbitrary sizes
// - Uses CUDA events to record H2D, kernel, and D2H timings
// Build: nvcc -O3 -arch=sm_60 vectoraddition.cu -o vectoraddition

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) do {                                \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
        int rtver = 0, drvver = 0;                          \
        cudaRuntimeGetVersion(&rtver);                      \
        cudaDriverGetVersion(&drvver);                      \
        const char *s = cudaGetErrorString(e);              \
        fprintf(stderr, "CUDA Error %s:%d: code=%d %s (rt=%d drv=%d)\n", \
                __FILE__, __LINE__, (int)e, s ? s : "(null)", rtver, drvver); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0)

// Kernel: element-wise C = A + B
__global__ void matrixAddKernel(const float *A, const float *B, float *C, int rows, int cols) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    for (int y = gy; y < rows; y += strideY) {
        int base = y * cols;
        for (int x = gx; x < cols; x += strideX) {
            int idx = base + x;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main(int argc, char **argv) {
    const int DEFAULT_N = 4096;
    const int rows = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    const int cols = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;
    const size_t N = (size_t)rows * (size_t)cols;

    printf("Matrix add %d x %d  (%.2fM elements)\n", rows, cols, N / 1e6f);

    // Diagnostic: report runtime/driver and device count/properties before allocations
    {
        int runtimeVer = 0, driverVer = 0;
        cudaRuntimeGetVersion(&runtimeVer);
        cudaDriverGetVersion(&driverVer);
        int devCount = 0;
        cudaError_t e = cudaGetDeviceCount(&devCount);
        const char *errstr = (e == cudaSuccess) ? "OK" : cudaGetErrorString(e);
        printf("CUDA runtime=%d driver=%d deviceCount=%d (getDeviceCount -> %d: %s)\n",
               runtimeVer, driverVer, devCount, (int)e, errstr ? errstr : "(null)");
        if (devCount <= 0) {
            fprintf(stderr, "No CUDA devices found or driver/runtime mismatch. Aborting.\n");
            return 1;
        }
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("Using device 0: %s (compute %d.%d, totalMem=%zu MB)\n",
               prop.name, prop.major, prop.minor, (size_t)(prop.totalGlobalMem / (1024*1024)));
        CUDA_CHECK(cudaSetDevice(0));
    }

    // Allocate pinned host memory for faster transfers
    float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, N * sizeof(float)));

    // Init on host
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0f;           // simple pattern for easy verification
        h_B[i] = 2.0f;
    }

    // Device buffers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Create events for timing H2D, kernel, and D2H
    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_kernel_end, ev_d2h_end;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_end));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_end));

    // Record H2D start
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice, 0));
    // Record H2D end after copies are issued (will be marked when copies complete)
    CUDA_CHECK(cudaEventRecord(ev_h2d_end, 0));

    // Kernel launch parameters
    dim3 block(32, 8); // 256 threads per block (good starting point)
    dim3 grid( (cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y );

    // Launch kernel (after H2D completes automatically due to same stream 0 ordering)
    matrixAddKernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    // Record kernel end (will be marked when kernel finished)
    CUDA_CHECK(cudaEventRecord(ev_kernel_end, 0));

    // Copy result back (D2H)
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end, 0));

    // Wait for all to complete
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_end));

    // Measure times
    float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_h2d_start, ev_h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev_h2d_end, ev_kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_kernel_end, ev_d2h_end));

    printf("Timings (ms): H2D = %.3f, kernel = %.3f, D2H = %.3f, total = %.3f\n",
           ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);

    // Verify result (simple)
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float want = h_A[i] + h_B[i];
        if (h_C[i] != want) {
            fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, h_C[i], want);
            ok = false;
            break;
        }
    }
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_end));
    CUDA_CHECK(cudaEventDestroy(ev_kernel_end));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_end));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return ok ? 0 : 1;
}