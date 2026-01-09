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

// External CUDA function
extern "C" {
    void launchVideoFilter(const uchar4* d_input, uchar4* d_output, int w, int h, int filter, int radius);
}

int main() {
    // Set stdin and stdout to binary mode
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    // 2. Allocate Memory
    size_t imgSize = VID_W * VID_H * sizeof(uchar4);

    std::vector<uchar4> h_videoBuffer(VID_W * VID_H);
    std::vector<uchar4> h_outputBuffer(VID_W * VID_H);
    
    uchar4* d_videoBuffer;
    uchar4* d_outputBuffer;

    if (cudaMalloc(&d_videoBuffer, imgSize) != cudaSuccess) return -1;
    if (cudaMalloc(&d_outputBuffer, imgSize) != cudaSuccess) return -1;

    // 3. Main Loop
    while (true) {
        // A. Read Filter Command from STDIN
        std::string filterLine;
        if (!std::getline(std::cin, filterLine)) break;
        int filter = 0;
        int radius = 0;
        if (filterLine.find("FILTER:") == 0) {
            size_t colon1 = filterLine.find(':');
            size_t colon2 = filterLine.find(':', colon1 + 1);
            if (colon2 != std::string::npos) {
                filter = std::stoi(filterLine.substr(colon1 + 1, colon2 - colon1 - 1));
                radius = std::stoi(filterLine.substr(colon2 + 1));
            } else {
                filter = std::stoi(filterLine.substr(colon1 + 1));
            }
        }

        // B. Read Video Frame from STDIN (Blocking)
        std::cin.read(reinterpret_cast<char*>(h_videoBuffer.data()), imgSize);
        if (std::cin.gcount() != imgSize) break;

        // C. Copy to GPU
        cudaMemcpy(d_videoBuffer, h_videoBuffer.data(), imgSize, cudaMemcpyHostToDevice);

        // D. Process on GPU
        launchVideoFilter(d_videoBuffer, d_outputBuffer, VID_W, VID_H, filter, radius);

        // E. Copy Result Back
        cudaMemcpy(h_outputBuffer.data(), d_outputBuffer, imgSize, cudaMemcpyDeviceToHost);

        // F. Send processed frame to stdout
        std::cout.write(reinterpret_cast<char*>(h_outputBuffer.data()), imgSize);
        std::cout.flush();
    }

    cudaFree(d_videoBuffer);
    cudaFree(d_outputBuffer);

    return 0;
}
