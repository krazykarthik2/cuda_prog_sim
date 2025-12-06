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
#define SAND 1
#define WALL 2

// External CUDA functions
extern "C" {
    void initGrid(int* grid, int w, int h);
    void stepSand(int* grid, int w, int h, int frame);
    void renderSand(int* grid, unsigned char* fb, int w, int h);
    void paintCircle(int* grid, int w, int h, int cx, int cy, int r, int type);
    void paintRect(int* grid, int w, int h, int rx, int ry, int rw, int rh, int type);
}

// UI Constants
const int SIM_W = 800;
const int SIM_H = 800;
const int UI_W = 220; // Widen slightly for controls
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

int main(){
    InitWindow(WIN_W, WIN_H, "CUDA Sand Solver");
    SetTargetFPS(600);

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
                    paintCircle(d_grid, SIM_W, SIM_H, (int)mouse.x, (int)mouse.y, ui.brushSize, SAND);
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
                stepSand(d_grid, SIM_W, SIM_H, frame++);
            }
        }

        // Render
        renderSand(d_grid, d_fb, SIM_W, SIM_H);
        cudaMemcpy(h_fb, d_fb, fbSize, cudaMemcpyDeviceToHost);
        UpdateTexture(tex, h_fb);

        BeginDrawing();
        ClearBackground(GetColor(0x181818FF)); // Sleek dark bg

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
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "DRAW SAND", ui.currentTool == TOOL_DRAW)) ui.currentTool = TOOL_DRAW;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "ERASER", ui.currentTool == TOOL_ERASER)) ui.currentTool = TOOL_ERASER;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "RECTANGLE", ui.currentTool == TOOL_RECT)) ui.currentTool = TOOL_RECT;
        by += gap;
        if (DrawButton({(float)bx, (float)by, (float)bw, (float)bh}, "CIRCLE", ui.currentTool == TOOL_CIRCLE)) ui.currentTool = TOOL_CIRCLE;
        by += gap + 15;

        // --- Brush Size ---
        DrawText("BRUSH SIZE", bx, by, 20, LIGHTGRAY); by += 30;
        // [-] Value [+]
        if (DrawButton({(float)bx, (float)by, 50, 40}, "-", false)) {
            if(ui.brushSize > 1) ui.brushSize--;
        }
        
        const char* szText = TextFormat("%d", ui.brushSize);
        int txW = MeasureText(szText, 20);
        DrawText(szText, bx + 50 + (90-txW)/2, by + 10, 20, WHITE);
        
        if (DrawButton({(float)bx + 140, (float)by, 50, 40}, "+", false)) {
            if(ui.brushSize < 100) ui.brushSize++;
        }
        by += gap + 10;

        // --- Speed ---
        DrawText("SPEED (Steps/Frame)", bx, by, 20, LIGHTGRAY); by += 30;
        if (DrawButton({(float)bx, (float)by, 50, 40}, "-", false)) {
            if(ui.stepsPerFrame > 1) ui.stepsPerFrame--;
        }

        const char* spText = TextFormat("%dx", ui.stepsPerFrame);
        int spW = MeasureText(spText, 20);
        DrawText(spText, bx + 50 + (90-spW)/2, by + 10, 20, WHITE);

        if (DrawButton({(float)bx + 140, (float)by, 50, 40}, "+", false)) {
            if(ui.stepsPerFrame < 50) ui.stepsPerFrame++;
        }
        by += gap;

        EndDrawing();
    }

    cudaFree(d_grid);
    cudaFree(d_fb);
    free(h_fb);
    CloseWindow();
    return 0;
}
