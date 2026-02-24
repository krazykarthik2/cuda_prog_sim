#include "raylib.h"
#include "rlgl.h"
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>

enum DitherMode {
    ORDERED_2x2,
    ORDERED_4x4,
    ORDERED_8x8,
    BLUE_NOISE,
    COUNT
};

const char* modeNames[] = {
    "Ordered 2x2",
    "Ordered 4x4",
    "Ordered 8x8",
    "Blue Noise"
};

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 840;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screenWidth, screenHeight, "GPU Dithering Playground (GLSL Shaders)");
    SetTargetFPS(60);
    srand(time(NULL));

    // 1. Load Image
    printf("Loading image...\n");
    Image image = LoadImage("sample.png");
    if (image.data == NULL) {
        printf("sample.png not found, generating gradient fallback...\n");
        image = GenImageGradientLinear(600, 600, 0, WHITE, BLACK);
    }
    
    Texture2D originalTex = LoadTextureFromImage(image);
    int imgW = image.width;
    int imgH = image.height;

    // 2. Load Shaders
    printf("Loading GLSL Shaders...\n");
    Shader ditherShader = LoadShader(0, "dither.frag");
    
    int modeLoc = GetShaderLocation(ditherShader, "ditherMode");
    int resLoc = GetShaderLocation(ditherShader, "resolution");
    int noiseLoc = GetShaderLocation(ditherShader, "noiseTex");

    Vector2 resolution = { (float)imgW, (float)imgH };
    SetShaderValue(ditherShader, resLoc, &resolution, SHADER_UNIFORM_VEC2);

    // 3. Generate Noise Texture for Blue Noise
    Image noiseImage = GenImageColor(64, 64, BLACK);
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            unsigned char val = rand() % 255;
            Color c = { val, val, val, 255 };
            ImageDrawPixel(&noiseImage, x, y, c);
        }
    }
    Texture2D noiseTex = LoadTextureFromImage(noiseImage);
    UnloadImage(noiseImage);

    DitherMode currentMode = ORDERED_8x8;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_RIGHT)) currentMode = (DitherMode)((currentMode + 1) % COUNT);
        if (IsKeyPressed(KEY_LEFT)) currentMode = (DitherMode)((currentMode - 1 + COUNT) % COUNT);

        // Update Shader Uniforms
        int mode = (int)currentMode;
        SetShaderValue(ditherShader, modeLoc, &mode, SHADER_UNIFORM_INT);

        BeginDrawing();
            ClearBackground(DARKGRAY);
            
            // Draw Original
            DrawText("Original", 50, 80, 20, RAYWHITE);
            DrawTexture(originalTex, 50, 120, WHITE);
            
            // Draw Dithered with Shader
            DrawText("GPU Dithered (GLSL)", 650, 80, 20, RAYWHITE);
            
            BeginShaderMode(ditherShader);
                // Bind the noise texture to the extra sampler
                rlActiveTextureSlot(1);
                rlEnableTexture(noiseTex.id);
                rlActiveTextureSlot(0); // Switch back to default slot for the main texture
                
                DrawTexture(originalTex, 650, 120, WHITE);
            EndShaderMode();

            DrawText("GPU Dithering Playground", 20, 20, 30, RAYWHITE);
            DrawText(TextFormat("Current Mode: %s", modeNames[currentMode]), 20, 60, 20, YELLOW);
            DrawText("STATUS: GPU Acceleration ACTIVE (GLSL Fragment Shader)", 20, 750, 20, GREEN);
            DrawText("Use LEFT/RIGHT arrows to switch modes", 20, 780, 20, LIGHTGRAY);
            
            DrawFPS(1100, 20);
        EndDrawing();
    }

    // Cleanup
    UnloadShader(ditherShader);
    UnloadTexture(originalTex);
    UnloadTexture(noiseTex);
    UnloadImage(image);
    CloseWindow();

    return 0;
}
