#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

struct Boid {
    float3 pos;
    float3 vel;
};

__device__ float3 op_add(float3 a, float3 b){ return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ float3 op_sub(float3 a, float3 b){ return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ float3 op_mul(float3 a, float b){ return make_float3(a.x*b,a.y*b,a.z*b); }

__device__ float3 normalize3(float3 a){
    float l = rsqrtf(a.x*a.x + a.y*a.y + a.z*a.z + 1e-6f);
    return make_float3(a.x*l, a.y*l, a.z*l);
}

__global__ void updateBoids(Boid* b, int n, float dt, float3 mouseAttractor,
                            float alignW, float cohW, float sepW, float attractW,
                            float boxSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    float3 pos = b[i].pos;
    float3 vel = b[i].vel;

    float3 align = make_float3(0,0,0);
    float3 coh   = make_float3(0,0,0);
    float3 sep   = make_float3(0,0,0);

    int count = 0;
    for(int j=0;j<n;j++){
        if(j==i) continue;
        float3 d = op_sub(b[j].pos, pos);
        float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;
        if(dist2 < 5.0f){
            align = op_add(align, b[j].vel);
            coh   = op_add(coh, b[j].pos);
            sep   = op_sub(sep, b[j].pos);
            count++;
        }
    }

    if(count > 0){
        align = op_mul(align, 1.0f/count);
        coh   = op_mul(coh,   1.0f/count);
        coh   = op_sub(coh, pos);
        sep   = op_mul(sep, -1.0f/count);
    }

    float3 attract = op_sub(mouseAttractor, pos);

    vel = op_add(vel,
        op_mul(normalize3(align), alignW*dt));
    vel = op_add(vel,
        op_mul(normalize3(coh),   cohW*dt));
    vel = op_add(vel,
        op_mul(normalize3(sep),   sepW*dt));
    vel = op_add(vel,
        op_mul(normalize3(attract), attractW*dt));

    // Soft boundary avoidance: apply a steering force when approaching the box edges
    float margin = boxSize*0.81f; // start steering when inside 81% of box
    float borderStrength = 6.0f; // tuning parameter
    if(pos.x > margin) vel.x += -borderStrength * (pos.x - margin) * dt;
    if(pos.x < -margin) vel.x +=  borderStrength * (-margin - pos.x) * dt;
    if(pos.y > margin) vel.y += -borderStrength * (pos.y - margin) * dt;
    if(pos.y < -margin) vel.y +=  borderStrength * (-margin - pos.y) * dt;
    if(pos.z > margin) vel.z += -borderStrength * (pos.z - margin) * dt;
    if(pos.z < -margin) vel.z +=  borderStrength * (-margin - pos.z) * dt;

    pos = op_add(pos, op_mul(vel, dt));

    // Keep boids inside the box by clamping small overshoots
    if(pos.x > boxSize) pos.x = boxSize;
    if(pos.x < -boxSize) pos.x = -boxSize;
    if(pos.y > boxSize) pos.y = boxSize;
    if(pos.y < -boxSize) pos.y = -boxSize;
    if(pos.z > boxSize) pos.z = boxSize;
    if(pos.z < -boxSize) pos.z = -boxSize;

    // Limit maximum speed
    float maxSpeed = 8.0f;
    float spd2 = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
    if(spd2 > maxSpeed*maxSpeed){
        float inv = rsqrtf(spd2 + 1e-9f);
        vel.x = vel.x * inv * maxSpeed;
        vel.y = vel.y * inv * maxSpeed;
        vel.z = vel.z * inv * maxSpeed;
    }

    b[i].pos = pos;
    b[i].vel = vel;
}

__global__ void clearFB(unsigned char* fb, int w, int h){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id >= w*h) return;
    fb[id*4+0] = 10;
    fb[id*4+1] = 10;
    fb[id*4+2] = 20;
    fb[id*4+3] = 255;
}

__global__ void rasterBoids(Boid* b, int n, unsigned char* fb,
                            int w, int h, float boidSize, float boxSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    float3 p = b[i].pos;

    // Ensure boid position is clamped to simulation box (safety)
    if(p.x > boxSize) p.x = boxSize;
    if(p.x < -boxSize) p.x = -boxSize;
    if(p.y > boxSize) p.y = boxSize;
    if(p.y < -boxSize) p.y = -boxSize;
    if(p.z > boxSize) p.z = boxSize;
    if(p.z < -boxSize) p.z = -boxSize;

    // Map world position to screen using boxSize so [-boxSize,boxSize] -> [0,w]/[0,h]
    float half = boxSize * 2.0f; // full span
    float nx = (p.x / half) + 0.5f; // normalized 0..1
    float ny = 0.5f - (p.y / half); // invert Y so +Y is up

    // clamp normalized coords to [0,1] to avoid rounding outside
    if(nx < 0.0f) nx = 0.0f; if(nx > 1.0f) nx = 1.0f;
    if(ny < 0.0f) ny = 0.0f; if(ny > 1.0f) ny = 1.0f;

    float x = nx * w;
    float y = ny * h;

    // Direction from velocity, fallback to (1,0)
    float3 v = b[i].vel;
    float vx = v.x, vy = v.y;
    float vlen = sqrtf(vx*vx + vy*vy) + 1e-9f;
    float dirx = vx / vlen;
    float diry = vy / vlen;
    float perp_x = -diry;
    float perp_y = dirx;

    // Arrow size in pixels
    float tip = boidSize * 4.0f;
    float back = boidSize * 1.0f;
    float halfWidth = boidSize * 1.6f;

    // rasterize a small bounding box around the boid
    int minx = (int)floorf(x - tip - halfWidth);
    int maxx = (int)ceilf(x + tip + halfWidth);
    int miny = (int)floorf(y - tip - halfWidth);
    int maxy = (int)ceilf(y + tip + halfWidth);

    if(minx < 0) minx = 0; if(miny < 0) miny = 0;
    if(maxx >= w) maxx = w-1; if(maxy >= h) maxy = h-1;

    for(int yy = miny; yy <= maxy; yy++){
        for(int xx = minx; xx <= maxx; xx++){
            // pixel center relative to boid
            float rx = (xx + 0.5f) - x;
            float ry = (yy + 0.5f) - y;

            // project onto dir/perp
            float dotDir = rx*dirx + ry*diry;
            float dotPerp = rx*perp_x + ry*perp_y;

            // check triangle shape: wider at back, narrow at tip
            if(dotDir >= -back && dotDir <= tip){
                float tnorm = (dotDir + back) / (tip + back); // 0 at back, 1 at tip
                if(tnorm < 0.0f) tnorm = 0.0f; if(tnorm > 1.0f) tnorm = 1.0f;
                float allowed = halfWidth * (1.0f - tnorm);
                if(fabsf(dotPerp) <= allowed){
                    int idx = (yy*w + xx)*4;
                    // simple shading: tip brighter
                    unsigned char r = (unsigned char)(180 + 75*tnorm);
                    unsigned char g = (unsigned char)(200 + 55*tnorm);
                    unsigned char bcol = (unsigned char)(255);
                    fb[idx+0] = r;
                    fb[idx+1] = g;
                    fb[idx+2] = bcol;
                    fb[idx+3] = 255;
                }
            }
        }
    }
}

__host__ void stepBoids(Boid* d_boids, int n, float dt, float3 attract,
                        float aW, float cW, float sW, float attractW, float boxSize)
{
    dim3 b(128), g((n+127)/128);
    updateBoids<<<g,b>>>(d_boids,n,dt,attract,aW,cW,sW,attractW,boxSize);
}

__host__ void drawBoids(Boid* d_boids, int n, unsigned char* d_fb,
                        int w,int h,float boidSize, float boxSize)
{
    dim3 b2(256), g2((w*h+255)/256);
    clearFB<<<g2,b2>>>(d_fb,w,h);

    dim3 b3(128), g3((n+127)/128);
    rasterBoids<<<g3,b3>>>(d_boids,n,d_fb,w,h,boidSize, boxSize);
}

}
