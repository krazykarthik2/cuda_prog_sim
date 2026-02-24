#include "raylib.h"
#include "raymath.h"
#include <iostream>

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Voxel Ray Tracer (GLSL)");
    DisableCursor();
    SetTargetFPS(60);

    Shader voxelShader = LoadShader(0, "voxels.frag");

    int camPosLoc = GetShaderLocation(voxelShader, "cameraPos");
    int camDirLoc = GetShaderLocation(voxelShader, "cameraDir");
    int camUpLoc = GetShaderLocation(voxelShader, "cameraUp");
    int resLoc = GetShaderLocation(voxelShader, "resolution");
    int timeLoc = GetShaderLocation(voxelShader, "time");

    Camera3D camera = { 0 };
    camera.position = { 10.0f, 20.0f, 10.0f };
    camera.target = { 15.0f, 15.0f, 15.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // 2. Setup Render Texture for Low-Res Rendering (Performance Boost)
    const int renderWidth = screenWidth / 2;
    const int renderHeight = screenHeight / 2;
    RenderTexture2D target = LoadRenderTexture(renderWidth, renderHeight);
    SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);

    Vector2 renderResolution = { (float)renderWidth, (float)renderHeight };
    SetShaderValue(voxelShader, resLoc, &renderResolution, SHADER_UNIFORM_VEC2);

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_FIRST_PERSON);

        float time = (float)GetTime();
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));

        SetShaderValue(voxelShader, camPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camDirLoc, &forward, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camUpLoc, &camera.up, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, timeLoc, &time, SHADER_UNIFORM_FLOAT);

        // 3. Render at Low Resolution
        BeginTextureMode(target);
            ClearBackground(BLACK);
            BeginShaderMode(voxelShader);
                // Draw a rectangle covering the entire low-res texture
                DrawRectangle(0, 0, renderWidth, renderHeight, WHITE);
            EndShaderMode();
        EndTextureMode();

        // 4. Upscale to Screen
        BeginDrawing();
            ClearBackground(DARKGRAY);
            
            // Draw the render texture to the full window size
            DrawTexturePro(target.texture, 
                { 0, 0, (float)target.texture.width, (float)-target.texture.height },
                { 0, 0, (float)screenWidth, (float)screenHeight },
                { 0, 0 }, 0.0f, WHITE);

            DrawText("OPTIMIZED: 50% Internal Resolution", 10, 10, 20, GREEN);
            DrawText("WASD + Mouse to Move", 10, 40, 20, RAYWHITE);
            DrawFPS(1100, 10);
        EndDrawing();
    }

    UnloadRenderTexture(target);

    UnloadShader(voxelShader);
    CloseWindow();

    return 0;
}
