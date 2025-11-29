#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ADD THESE float3 OPERATORS (CUDA does not define them)
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(const float3 &a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float3 operator*(float b, const float3 &a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}


extern "C" {
    // produce an RGBA8 image into host pointer (pinned)
    cudaError_t generate_image(unsigned char* dst, int width, int height, float time);
}

// simple signed distance function (sphere + floor) for raymarching
__device__ float sdSphere(float3 p, float r){ return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) - r; }
__device__ float sdPlane(float3 p, float y){ return p.y - y; }

__device__ float mapScene(float3 p){
    float dSphere = sdSphere(p - make_float3(0,1.0f,6.0f), 1.0f);
    float dPlane = sdPlane(p, 0.0f);
    return fminf(dSphere, dPlane);
}

__device__ float3 estimateNormal(float3 p){
    const float eps = 1e-3f;
    float3 n;
    n.x = mapScene(p + make_float3(eps,0,0)) - mapScene(p - make_float3(eps,0,0));
    n.y = mapScene(p + make_float3(0,eps,0)) - mapScene(p - make_float3(0,eps,0));
    n.z = mapScene(p + make_float3(0,0,eps)) - mapScene(p - make_float3(0,0,eps));
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z) + 1e-9f;
    return make_float3(n.x/len, n.y/len, n.z/len);
}

__device__ float traceDistance(float3 ro, float3 rd){
    float t = 0.0f;
    for(int i=0;i<128;i++){
        float3 p = ro + rd * t;
        float d = mapScene(p);
        if(d < 1e-3f) return t;
        t += d;
        if(t > 100.0f) break;
    }
    return 1e9f;
}

__global__ void renderKernel(unsigned char* img, int width, int height, float time){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=width || y>=height) return;
    int idx = (y*width + x)*4;

    // normalised coords
    float u = (2.0f * (x + 0.5f) / width - 1.0f) * (width/(float)height);
    float v = (1.0f - 2.0f * (y + 0.5f) / height);

    float3 ro = make_float3(0.0f, 1.0f, 0.0f);
    float3 rd = make_float3(u, v, 1.0f);
    // normalize
    float rl = sqrtf(rd.x*rd.x + rd.y*rd.y + rd.z*rd.z);
    rd.x/=rl; rd.y/=rl; rd.z/=rl;

    float tdist = traceDistance(ro, rd);
    unsigned char r=20, g=30, b=40, a=255;
    if(tdist < 1e8f){
        float3 p = ro + rd * tdist;
        float3 n = estimateNormal(p);
        // simple diffuse with moving light
        float3 light = make_float3(sinf(time)*4.0f, 4.0f + cosf(time), 2.0f);
        float3 L = light - p;
        float llen = sqrtf(L.x*L.x + L.y*L.y + L.z*L.z) + 1e-9f;
        L.x/=llen; L.y/=llen; L.z/=llen;
        float diff = fmaxf(0.0f, n.x*L.x + n.y*L.y + n.z*L.z);
        float shade = diff;
        r = (unsigned char)fminf(255.0f, 120.0f * shade + 50.0f);
        g = (unsigned char)fminf(255.0f, 200.0f * shade + 60.0f);
        b = (unsigned char)fminf(255.0f, 150.0f * shade + 80.0f);
    }

    img[idx+0] = r; // R
    img[idx+1] = g; // G
    img[idx+2] = b; // B
    img[idx+3] = a; // A
}

extern "C" cudaError_t generate_image(unsigned char* dst, int width, int height, float time){
    // dst is host pinned memory (RGBA8). We'll allocate device buffer and launch kernel.
    unsigned char* d_img = nullptr;
    size_t bufBytes = (size_t)width * height * 4;
    cudaError_t err = cudaMalloc(&d_img, bufBytes);
    if(err != cudaSuccess) return err;

    dim3 block(16,16);
    dim3 grid((width + block.x -1)/block.x, (height + block.y -1)/block.y);
    renderKernel<<<grid, block>>>(d_img, width, height, time);
    err = cudaGetLastError();
    if(err != cudaSuccess){ cudaFree(d_img); return err; }

    err = cudaMemcpy(dst, d_img, bufBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    return err;
}
