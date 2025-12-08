#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>

// Video Dimensions (Must match the Python sender)
const int VID_W = 1280;
const int VID_H = 720;

// Terminal Output Dimensions (Downsampled)
#define SCALE .1
const int TERM_W = (int)(VID_W * SCALE);
const int TERM_H = (int)(VID_H * SCALE);

// External CUDA function
extern "C" {
    void launchVideoToAscii(const uchar4* d_input, char* d_output, int in_w, int in_h, int out_w, int out_h);
}

int main() {
    // 1. Setup Windows Console
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= 0x0004; // ENABLE_VIRTUAL_TERMINAL_PROCESSING
    SetConsoleMode(hOut, dwMode);

    // Set stdin to binary mode
    _setmode(_fileno(stdin), _O_BINARY);

    // 2. Allocate Memory
    size_t imgSize = VID_W * VID_H * sizeof(uchar4);
    size_t asciiSize = TERM_W * TERM_H * sizeof(char);

    // Use pinned memory for faster Host<->Device transfers if possible, 
    // but standard new/malloc is okay for this bandwidth. 
    // cudaHostAlloc is better but let's stick to vector for simplicity unless requested.
    std::vector<uchar4> h_videoBuffer(VID_W * VID_H);
    std::vector<char> h_asciiBuffer(TERM_W * TERM_H);
    
    // Output buffer for fwrite. Size = (Width + Newline) * Height
    std::vector<char> h_displayBuffer((TERM_W + 1) * TERM_H);

    uchar4* d_videoBuffer;
    char* d_asciiBuffer;

    if (cudaMalloc(&d_videoBuffer, imgSize) != cudaSuccess) return -1;
    if (cudaMalloc(&d_asciiBuffer, asciiSize) != cudaSuccess) return -1;

    // 3. Main Loop
    system("cls");
    printf("\033[?25l"); // Hide cursor

    // Constants for copy loop
    const int rowStride = TERM_W + 1;

    while (true) {
        // A. Read Video Frame from STDIN (Blocking)
        // Note: Check for end of stream
        std::cin.read(reinterpret_cast<char*>(h_videoBuffer.data()), imgSize);
        if (std::cin.gcount() != imgSize) break;

        // B. Copy to GPU
        cudaMemcpy(d_videoBuffer, h_videoBuffer.data(), imgSize, cudaMemcpyHostToDevice);

        // C. Process on GPU
        launchVideoToAscii(d_videoBuffer, d_asciiBuffer, VID_W, VID_H, TERM_W, TERM_H);

        // D. Copy Result Back
        cudaMemcpy(h_asciiBuffer.data(), d_asciiBuffer, asciiSize, cudaMemcpyDeviceToHost);

        // E. Format output efficiently
        // We construct the display buffer directly to avoid std::string allocations
        char* src = h_asciiBuffer.data();
        char* dst = h_displayBuffer.data();
        
        // Parallelizing this loop on CPU is likely overkill for 90 iterations, 
        // but memcpy is very fast.
        for (int y = 0; y < TERM_H; ++y) {
            memcpy(dst + y * rowStride, src + y * TERM_W, TERM_W);
            (dst + y * rowStride)[TERM_W] = '\n';
        }

        // F. Display
        // Move cursor via ANSI
        printf("\033[H"); 
        
        // Write entire buffer in one syscall
        // Note: fwrite is generally buffered, but full block write is efficient.
        fwrite(h_displayBuffer.data(), 1, h_displayBuffer.size(), stdout);
        
        // Flush not strictly necessary if we are continuously writing, 
        // but ensures frames don't lag in buffer.
        // fflush(stdout); 
    }

    // Restore
    printf("\033[?25h");
    cudaFree(d_videoBuffer);
    cudaFree(d_asciiBuffer);

    return 0;
}
