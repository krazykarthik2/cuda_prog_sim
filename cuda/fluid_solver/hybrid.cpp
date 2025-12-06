#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>
#include "raylib.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // For std::min/max

// CUDA Constants
#define EMPTY 0
#define WATER 1
#define WALL 2

// External CUDA functions
extern "C" {
    void initGrid(int* grid, int w, int h);
    void stepFluids(int* grid, int w, int h, int frame, int gravitySteps, int flowSteps);
    void renderFluids(int* grid, unsigned char* fb, int w, int h);
    void paintCircle(int* grid, int w, int h, int cx, int cy, int r, int type);
    void paintRect(int* grid, int w, int h, int rx, int ry, int rw, int rh, int type);
}

// UI Constants
const int SIM_W = 800;
const int SIM_H = 800;
const int UI_W = 220; 
const int WIN_W = SIM_W + UI_W;
const int WIN_H = SIM_H;

enum Tool {
    TOOL_DRAW,
    TOOL_RECT,
    TOOL_CIRCLE,
    TOOL_ERASER
};

struct UIState {
    bool running = true;
    Tool currentTool = TOOL_DRAW;
    int brushSize = 5;
    int stepsPerFrame = 1;
    
    // New parameters 
    int gravitySteps = 1;
    int flowSteps = 8;
    
    // For drag shapes
    bool isDragging = false;
    Vector2 dragStart = {0};
};

bool DrawButton(Rectangle rect, const char* text, bool active) {
    bool clicked = false;
    Vector2 mouse = GetMousePosition();
    bool hover = CheckCollisionPointRec(mouse, rect);
    
    Color col = active ? SKYBLUE : (hover ? LIGHTGRAY : GRAY);
    if (active) col = BLUE;
    
    DrawRectangleRec(rect, col);
    DrawRectangleLinesEx(rect, 2, BLACK);
    
    int fontSize = 20;
    int textW = MeasureText(text, fontSize);
    DrawText(text, (int)rect.x + ((int)rect.width - textW)/2, (int)rect.y + ((int)rect.height - fontSize)/2, fontSize, WHITE);
    
    if (hover && IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
        clicked = true;
    }
    return clicked;
}

void DrawControl(int x, int y, const char* label, int* value, int minVal, int maxVal) {
    DrawText(label, x, y, 20, LIGHTGRAY); 
    y += 30;
    
    if (DrawButton({(float)x, (float)y, 50, 40}, "-", false)) {
        if(*value > minVal) (*value)--;
    }
    
    const char* tx = TextFormat("%d", *value);
    int txW = MeasureText(tx, 20);
    DrawText(tx, x + 50 + (90-txW)/2, y + 10, 20, WHITE);
    
    if (DrawButton({(float)x + 140, (float)y, 50, 40}, "+", false)) {
        if(*value < maxVal) (*value)++;
    }
}

int main(){
    InitWindow(WIN_W, WIN_H, "CUDA Fluid Solver");
    SetTargetFPS(60);

    // Allocate GPU memory
    int* d_grid;
    unsigned char* d_fb;
    unsigned char* h_fb = (unsigned char*)malloc(SIM_W*SIM_H*4);

    size_t gridSize = SIM_W * SIM_H * sizeof(int);
    size_t fbSize = SIM_W * SIM_H * 4;

    cudaMalloc(&d_grid, gridSize);
    cudaMalloc(&d_fb, fbSize);

    initGrid(d_grid, SIM_W, SIM_H);

    Texture2D tex = LoadTextureFromImage({h_fb, SIM_W, SIM_H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8});

    UIState ui;
    int frame = 0;

    while(!WindowShouldClose()){
        // Input Handling
        Vector2 mouse = GetMousePosition();
        bool inSim = mouse.x < SIM_W;
        
        // Sim Interaction
        if (inSim) {
            if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
                if (ui.currentTool == TOOL_DRAW) {
                    paintCircle(d_grid, SIM_W, SIM_H, (int)mouse.x, (int)mouse.y, ui.brushSize, WATER);
                } else if (ui.currentTool == TOOL_ERASER) {
                    paintCircle(d_grid, SIM_W, SIM_H, (int)mouse.x, (int)mouse.y, ui.brushSize * 2, EMPTY);
                } else if (!ui.isDragging && (ui.currentTool == TOOL_RECT || ui.currentTool == TOOL_CIRCLE)) {
                    ui.isDragging = true;
                    ui.dragStart = mouse;
                }
            }
        }

        if (ui.isDragging) {
            if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
                // Commit shape
                if (ui.currentTool == TOOL_RECT) {
                    int x = (int)std::min(ui.dragStart.x, mouse.x);
                    int y = (int)std::min(ui.dragStart.y, mouse.y);
                    int w = (int)std::abs(mouse.x - ui.dragStart.x);
                    int h = (int)std::abs(mouse.y - ui.dragStart.y);
                    paintRect(d_grid, SIM_W, SIM_H, x, y, w, h, WALL);
                } else if (ui.currentTool == TOOL_CIRCLE) {
                    float dx = mouse.x - ui.dragStart.x;
                    float dy = mouse.y - ui.dragStart.y;
                    int r = (int)sqrtf(dx*dx + dy*dy);
                    paintCircle(d_grid, SIM_W, SIM_H, (int)ui.dragStart.x, (int)ui.dragStart.y, r, WALL);
                }
                ui.isDragging = false;
            }
        }
        
        // Handle Brush Size Scroll
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            ui.brushSize += (int)wheel;
            if (ui.brushSize < 1) ui.brushSize = 1;
            if (ui.brushSize > 50) ui.brushSize = 50;
        }

        // Sim Step
        if (ui.running) {
            for(int i=0; i<ui.stepsPerFrame; i++) {
                stepFluids(d_grid, SIM_W, SIM_H, frame++, ui.gravitySteps, ui.flowSteps);
            }
        }

        // Render
        renderFluids(d_grid, d_fb, SIM_W, SIM_H);
        cudaMemcpy(h_fb, d_fb, fbSize, cudaMemcpyDeviceToHost);
        UpdateTexture(tex, h_fb);

        BeginDrawing();
        ClearBackground(GetColor(0x181818FF)); 

        // Draw Simulation
        DrawTexture(tex, 0, 0, WHITE);

        // Draw Drag Preview
        if (ui.isDragging) {
            if (ui.currentTool == TOOL_RECT) {
                int x = (int)std::min(ui.dragStart.x, mouse.x);
                int y = (int)std::min(ui.dragStart.y, mouse.y);
                int w = (int)std::abs(mouse.x - ui.dragStart.x);
                int h = (int)std::abs(mouse.y - ui.dragStart.y);
                DrawRectangleLines(x, y, w, h, RED);
            } else if (ui.currentTool == TOOL_CIRCLE) {
                float dx = mouse.x - ui.dragStart.x;
                float dy = mouse.y - ui.dragStart.y;
                float r = sqrtf(dx*dx + dy*dy);
                DrawCircleLines((int)ui.dragStart.x, (int)ui.dragStart.y, r, RED);
            }
        }
        
        // Draw Brush Preview
        if (inSim && !ui.isDragging) {
             DrawCircleLines((int)mouse.x, (int)mouse.y, (float)ui.brushSize, GREEN);
        }

        // Draw UI
        int bx = SIM_W + 15;
        int by = 15;
        int bw = 190;
        int bh = 35;
        int gap = 45;

        // --- Controls ---
        DrawText("CONTROLS", bx, by, 20, LIGHTGRAY); by += 30;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, ui.running ? "PAUSE" : "RESUME", false)) {
            ui.running = !ui.running;
        }
        by += gap;

        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "CLEAR CANVAS", false)) {
            initGrid(d_grid, SIM_W, SIM_H);
        }
        by += gap + 15;

        // --- Tools ---
        DrawText("TOOLS", bx, by, 20, LIGHTGRAY); by += 30;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "DRAW WATER", ui.currentTool == TOOL_DRAW)) ui.currentTool = TOOL_DRAW;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "ERASER", ui.currentTool == TOOL_ERASER)) ui.currentTool = TOOL_ERASER;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "RECTANGLE", ui.currentTool == TOOL_RECT)) ui.currentTool = TOOL_RECT;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "CIRCLE", ui.currentTool == TOOL_CIRCLE)) ui.currentTool = TOOL_CIRCLE;
        by += gap + 15;

        // --- Sliders ---
        DrawControl(bx, by, "BRUSH SIZE", &ui.brushSize, 1, 100); by += gap + 20;
        DrawControl(bx, by, "GRAVITY STEPS", &ui.gravitySteps, 1, 10); by += gap + 20;
        DrawControl(bx, by, "FLOW STEPS", &ui.flowSteps, 1, 50); by += gap + 20;
        DrawControl(bx, by, "SIM SPEED", &ui.stepsPerFrame, 1, 20); by += gap + 20;


        EndDrawing();
    }

    cudaFree(d_grid);
    cudaFree(d_fb);
    free(h_fb);
    CloseWindow();
    return 0;
}
