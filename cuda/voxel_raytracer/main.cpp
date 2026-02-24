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
    camera.position = (Vector3){ 10.0f, 20.0f, 10.0f };
    camera.target = (Vector3){ 15.0f, 15.0f, 15.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Vector2 resolution = { (float)screenWidth, (float)screenHeight };
    SetShaderValue(voxelShader, resLoc, &resolution, SHADER_UNIFORM_VEC2);

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_FIRST_PERSON);

        float time = (float)GetTime();
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));

        SetShaderValue(voxelShader, camPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camDirLoc, &forward, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camUpLoc, &camera.up, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, timeLoc, &time, SHADER_UNIFORM_FLOAT);

        BeginDrawing();
            ClearBackground(BLACK);
            
            BeginShaderMode(voxelShader);
                DrawRectangle(0, 0, screenWidth, screenHeight, WHITE);
            EndShaderMode();

            DrawText("WASD + Mouse to Move", 10, 10, 20, RAYWHITE);
            DrawFPS(1100, 10);
        EndDrawing();
    }

    UnloadShader(voxelShader);
    CloseWindow();

    return 0;
}
