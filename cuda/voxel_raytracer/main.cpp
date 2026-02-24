#include "raylib.h"
#include "raymath.h"
#include <iostream>

float GetWorldHeight(float x, float z) {
    return floorf(sinf(x * 0.1f) * 3.0f + cosf(z * 0.1f) * 3.0f + 8.0f);
}

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Voxel Ray Tracer - Physics & Cheats");
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
    camera.fovy = 50.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    const int renderWidth = screenWidth / 2;
    const int renderHeight = screenHeight / 2;
    RenderTexture2D target = LoadRenderTexture(renderWidth, renderHeight);
    SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);

    Vector2 renderResolution = { (float)renderWidth, (float)renderHeight };
    SetShaderValue(voxelShader, resLoc, &renderResolution, SHADER_UNIFORM_VEC2);

    float velocityY = 0.0f;
    bool isGrounded = false;
    bool flightMode = false;
    bool showCheatMenu = false;
    const float gravity = -20.0f;
    const float jumpPower = 9.0f;
    const float eyeHeight = 1.7f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        
        // --- INPUT & CHEATS ---
        if (IsKeyPressed(KEY_SLASH)) {
            showCheatMenu = !showCheatMenu;
            if (showCheatMenu) EnableCursor(); else DisableCursor();
        }
        if (showCheatMenu && IsKeyPressed(KEY_F)) flightMode = !flightMode;

        Vector3 oldPos = camera.position;
        UpdateCamera(&camera, CAMERA_FIRST_PERSON);

        if (flightMode) {
            if (IsKeyDown(KEY_SPACE)) camera.position.y += 10.0f * dt;
            if (IsKeyDown(KEY_LEFT_CONTROL)) camera.position.y -= 10.0f * dt;
            velocityY = 0;
        } else {
            // Apply Gravity
            velocityY += gravity * dt;
            camera.position.y += velocityY * dt;

            // Ground Detection (Current Position)
            float groundH = GetWorldHeight(camera.position.x, camera.position.z) + 1.0f;
            float floorLevel = groundH + eyeHeight;

            if (camera.position.y <= floorLevel) {
                camera.position.y = floorLevel;
                velocityY = 0;
                isGrounded = true;
            } else {
                isGrounded = false;
            }

            // Jumping
            if (isGrounded && IsKeyPressed(KEY_SPACE)) {
                velocityY = jumpPower;
                isGrounded = false;
            }

            // Step Climbing Logic (Move check)
            float hAtNewPos = GetWorldHeight(camera.position.x, camera.position.z) + 1.0f;
            float hAtOldPos = GetWorldHeight(oldPos.x, oldPos.z) + 1.0f;
            
            // 1-step auto-climb
            if (hAtNewPos > hAtOldPos && hAtNewPos <= hAtOldPos + 1.1f) {
                camera.position.y = hAtNewPos + eyeHeight;
            }
            // 2-step climb only if jumping (checked via vertical velocity or being in air)
            else if (hAtNewPos > hAtOldPos + 1.1f && hAtNewPos <= hAtOldPos + 2.1f) {
                if (velocityY > 2.0f) { // Only if we HAVE jumped recently
                    camera.position.y = hAtNewPos + eyeHeight;
                } else {
                    // Blocked
                    camera.position.x = oldPos.x;
                    camera.position.z = oldPos.z;
                }
            }
        }

        float time = (float)GetTime();
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));

        SetShaderValue(voxelShader, camPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camDirLoc, &forward, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, camUpLoc, &camera.up, SHADER_UNIFORM_VEC3);
        SetShaderValue(voxelShader, timeLoc, &time, SHADER_UNIFORM_FLOAT);

        BeginTextureMode(target);
            ClearBackground(BLACK);
            BeginShaderMode(voxelShader);
                DrawRectangle(0, 0, renderWidth, renderHeight, WHITE);
            EndShaderMode();
        EndTextureMode();

        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexturePro(target.texture, 
                { 0, 0, (float)target.texture.width, (float)-target.texture.height },
                { 0, 0, (float)screenWidth, (float)screenHeight },
                { 0, 0 }, 0.0f, WHITE);

            DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, GREEN);
            DrawText(flightMode ? "MODE: FLIGHT (CHEAT)" : "MODE: SURVIVAL", 10, 40, 20, YELLOW);
            DrawText("Press '/' for Cheat Menu", 10, 70, 20, RAYWHITE);

            if (showCheatMenu) {
                DrawRectangle(screenWidth/2 - 150, screenHeight/2 - 80, 300, 160, Fade(BLACK, 0.8f));
                DrawRectangleLines(screenWidth/2 - 150, screenHeight/2 - 80, 300, 160, RAYWHITE);
                DrawText("--- CHEATS ---", screenWidth/2 - 80, screenHeight/2 - 60, 20, GOLD);
                DrawText("[F] Toggle Flight Mode", screenWidth/2 - 130, screenHeight/2 - 20, 20, RAYWHITE);
                DrawText("[ESC] Close", screenWidth/2 - 130, screenHeight/2 + 20, 20, RAYWHITE);
            }
        EndDrawing();
    }

    UnloadRenderTexture(target);
    UnloadShader(voxelShader);
    CloseWindow();
    return 0;
}
