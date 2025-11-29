#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>
#include "raylib.h"
#include <iostream>
#include <cstdlib>

struct Boid { float3 pos; float3 vel; };

extern "C" void stepBoids(Boid*, int, float, float3, float,float,float,float,float);
extern "C" void drawBoids(Boid*, int, unsigned char*, int,int,float, float);

bool isMouseInWindow(){
    int mx = GetMouseX();
    int my = GetMouseY();
    return mx >= 0 && mx < GetScreenWidth() && my >=0 && my < GetScreenHeight();
}
int main(){
    const int W=1280, H=720;
    InitWindow(W,H,"CUDA 3D Boids (GPU-rendered)");

    SetTargetFPS(60);

    int n = 1024*32;
    float box = 20.f;
    float boidSize = 3.f;

    float alignW=1.5f, cohW=1.0f, sepW=2.0f,attractW=5.0f;

    Boid* d_boids;
    cudaError_t cerr = cudaMalloc((void**)&d_boids, n*sizeof(Boid));
    if(cerr != cudaSuccess){
        std::cerr << "cudaMalloc d_boids failed: " << cudaGetErrorString(cerr) << std::endl;
        return 1;
    }

    Boid* h_init = new Boid[n];
    for(int i=0;i<n;i++){
        h_init[i].pos = make_float3(
            (float(rand())/RAND_MAX-0.5f)*box,
            (float(rand())/RAND_MAX-0.5f)*box,
            (float(rand())/RAND_MAX-0.5f)*box
        );
        h_init[i].vel = make_float3(
            (float(rand())/RAND_MAX-0.5f)*2,
            (float(rand())/RAND_MAX-0.5f)*2,
            (float(rand())/RAND_MAX-0.5f)*2
        );
    }
    cudaMemcpy(d_boids,h_init,n*sizeof(Boid),cudaMemcpyHostToDevice);
    delete[] h_init;

    unsigned char* d_fb;
    cerr = cudaMalloc((void**)&d_fb, W*H*4);
    if(cerr != cudaSuccess){
        std::cerr << "cudaMalloc d_fb failed: " << cudaGetErrorString(cerr) << std::endl;
        cudaFree(d_boids);
        return 1;
    }

    unsigned char* h_fb = nullptr;
    cerr = cudaHostAlloc((void**)&h_fb, W*H*4, cudaHostAllocPortable);
    if(cerr != cudaSuccess){
        std::cerr << "cudaHostAlloc failed, falling back to malloc: " << cudaGetErrorString(cerr) << std::endl;
        h_fb = (unsigned char*)malloc(W*H*4);
        if(!h_fb){
            std::cerr << "malloc failed" << std::endl;
            cudaFree(d_boids);
            cudaFree(d_fb);
            return 1;
        }
    }

    Image img = {h_fb, W,H,1,PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    Texture2D tex = LoadTextureFromImage(img);

    while(!WindowShouldClose()){
        float dt = GetFrameTime();

        float mx = (float)GetMouseX()/W;
        float my = (float)GetMouseY()/H;
        float3 attract = make_float3(
            (mx-0.5f)*box*2,
            (0.5f-my)*box*2,
            0
        );

        boidSize += GetMouseWheelMove()*0.5f;
        if(boidSize < 1) boidSize = 1;

        bool shiftDown = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        if(IsKeyDown(KEY_ONE))    alignW += shiftDown ? -dt : dt;
        if(IsKeyDown(KEY_TWO))    cohW   += shiftDown ? -dt : dt;
        if(IsKeyDown(KEY_THREE))  sepW   += shiftDown ? -dt : dt;
        if(IsKeyDown(KEY_FOUR))   attractW += shiftDown ? -dt : dt;
        
        
        stepBoids(d_boids,n,dt,attract,alignW,cohW,sepW,isMouseInWindow()?attractW:0,box);
        drawBoids(d_boids,n,d_fb,W,H,boidSize, box);

        cudaMemcpy(h_fb, d_fb, W*H*4, cudaMemcpyDeviceToHost);
        UpdateTexture(tex, (const void*)h_fb);

        BeginDrawing();
        ClearBackground(BLACK);
        DrawTexture(tex,0,0,WHITE);

        DrawText(TextFormat("Mouse = attractor(%.2f, %.2f)", (mx-0.5f)*box*2, (0.5f-my)*box*2),10,10,20,WHITE);
        DrawText(TextFormat("BoidSize: %.1f (mouse wheel)",boidSize),10,40,20,WHITE);
        DrawText(TextFormat("[1]Align: %.2f [2]Coh: %.2f [3]Sep: %.2f [4]Attract: %.2f",alignW,cohW,sepW,attractW),10,70,20,WHITE);

        EndDrawing();
    }

    // cleanup
    UnloadTexture(tex);
    if(h_fb){
        // if allocated with cudaHostAlloc, free with cudaFreeHost, otherwise free()
        // We can try cudaFreeHost; if pointer wasn't registered, it will return an error, so guard with cudaHostAlloc check not available here.
        // Safer to call cudaFreeHost and if it fails, call free.
        if(cudaFreeHost(h_fb) != cudaSuccess) free(h_fb);
    }
    if(d_fb) cudaFree(d_fb);
    if(d_boids) cudaFree(d_boids);

    CloseWindow();
}
