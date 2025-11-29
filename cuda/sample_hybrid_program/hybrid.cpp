// Minimal raylib + CUDA interop example

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdlib>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cuda_runtime.h>

#include "raylib.h"

// declare external CUDA function (implemented in CUDA .cu file)
extern "C" cudaError_t generate_image(unsigned char* dst, int width, int height, float time);

int main(){
    const int width = 800;
    const int height = 600;

    // Init raylib window
    InitWindow(width, height, "raylib + CUDA renderer");
    SetTargetFPS(60);

    // Create texture that we will update from CUDA generated pixels
    Image img = { 0 };
    img.data = (void*)malloc(width*height*4);
    img.width = width; img.height = height; img.mipmaps = 1; img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

    Texture2D tex = LoadTextureFromImage(img);

    // allocate pinned host memory for faster cudaMemcpy
    unsigned char* hostBuf = nullptr;
    cudaError_t cerr = cudaHostAlloc((void**)&hostBuf, width*height*4, cudaHostAllocPortable);
    if(cerr != cudaSuccess){
        std::cerr << "cudaHostAlloc failed, falling back to malloc\n";
        hostBuf = (unsigned char*)malloc(width*height*4);
    }

    // Use raylib's GetTime() for timing to avoid pulling in Windows API symbols
    double tstart = GetTime();

    while(!WindowShouldClose()){
        float now = (float)(GetTime() - tstart);

        // call CUDA to generate into hostBuf
        cudaError_t e = generate_image(hostBuf, width, height, now);
        if(e != cudaSuccess){
            std::cerr << "CUDA generate_image failed: " << e << std::endl;
            // if CUDA failed, we still draw last image
        } else {
            // update raylib image data then update texture
            // Note: raylib makes its own copies for streaming textures; use UpdateTexture for efficiency.
            UpdateTexture(tex, (const void*)hostBuf);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawTexture(tex, 0, 0, WHITE);
        DrawFPS(10,10);
        EndDrawing();

        // small sleep to be nice on CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // cleanup
    UnloadTexture(tex);
    free(img.data);
    if(hostBuf) {
        // If host memory was allocated by cudaHostAlloc, cudaFreeHost will free it, otherwise free() is required.
        // cudaFreeHost will return error for pointers not allocated with cudaHostAlloc, but it's safe to call
        // only if cudaHostAlloc succeeded. We used `cerr` earlier to detect that.
        if(cerr == cudaSuccess) cudaFreeHost(hostBuf);
        else free(hostBuf);
    }
    CloseWindow();
    return 0;
}

