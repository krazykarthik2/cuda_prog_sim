#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform sampler2D noiseTex;
uniform int ditherMode; // 0: 2x2, 1: 4x4, 2: 8x8, 3: Blue Noise
uniform vec2 resolution;

// Output fragment color
out vec4 finalColor;

// Bayer Matrices
float GetBayerValue(int x, int y, int size) {
    if (size == 2) {
        float m[4] = float[](0.0, 0.5, 0.75, 0.25);
        return m[(y % 2) * 2 + (x % 2)];
    }
    if (size == 4) {
        float m[16] = float[](
            0.0, 0.5, 0.125, 0.625,
            0.75, 0.25, 0.875, 0.375,
            0.1875, 0.6875, 0.0625, 0.5625,
            0.9375, 0.4375, 0.8125, 0.3125
        );
        return m[(y % 4) * 4 + (x % 4)];
    }
    // Default 8x8 (simplified lookup for brevity, can be expanded)
    return 0.5; 
}

void main() {
    // Get pixel color from texture
    vec4 texelColor = texture(texture0, fragTexCoord);
    
    // Calculate luminance
    float luminance = (0.299 * texelColor.r + 0.587 * texelColor.g + 0.114 * texelColor.b);
    
    // Pixel coordinates
    ivec2 pixelCoord = ivec2(fragTexCoord * resolution);
    
    float threshold = 0.5;
    
    if (ditherMode == 0) threshold = GetBayerValue(pixelCoord.x, pixelCoord.y, 2);
    else if (ditherMode == 1) threshold = GetBayerValue(pixelCoord.x, pixelCoord.y, 4);
    else if (ditherMode == 2) {
        // Simple 8x8 bayer pattern approximation
        threshold = mod(float(pixelCoord.x * 3 + pixelCoord.y * 5), 8.0) / 8.0;
    }
    else if (ditherMode == 3) {
        threshold = texture(noiseTex, fragTexCoord * (resolution / 64.0)).r;
    }
    
    float val = (luminance > threshold) ? 1.0 : 0.0;
    finalColor = vec4(vec3(val), texelColor.a);
}
