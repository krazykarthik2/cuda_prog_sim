#version 330

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 cameraUp;
uniform vec2 resolution;
uniform float time;

const int MAX_STEPS = 120;
const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, -0.4));

// Simple procedural voxel field
bool getVoxel(ivec3 p) {
    if (p.y < 0) return true;
    if (p.y > 32) return false;
    
    float f = sin(p.x * 0.1) * cos(p.z * 0.1) * 8.0 + 10.0;
    return float(p.y) < f;
}

void main() {
    // Ray Setup
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / resolution.y;
    vec3 right = normalize(cross(cameraDir, cameraUp));
    vec3 up = normalize(cross(right, cameraDir));
    vec3 rd = normalize(cameraDir + uv.x * right + uv.y * up);
    vec3 ro = cameraPos;

    // Fast Voxel Traversal (DDA)
    ivec3 pos = ivec3(floor(ro));
    vec3 deltaDist = abs(1.0 / rd);
    ivec3 step = ivec3(sign(rd));
    vec3 sideDist = (sign(rd) * (vec3(pos) - ro) + (sign(rd) * 0.5) + 0.5) * deltaDist;

    int hitSide = 0; // 0:x, 1:y, 2:z
    bool hit = false;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        if (getVoxel(pos)) {
            hit = true;
            break;
        }
        
        if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
            sideDist.x += deltaDist.x;
            pos.x += step.x;
            hitSide = 0;
        } else if (sideDist.y < sideDist.z) {
            sideDist.y += deltaDist.y;
            pos.y += step.y;
            hitSide = 1;
        } else {
            sideDist.z += deltaDist.z;
            pos.z += step.z;
            hitSide = 2;
        }
    }

    if (hit) {
        // Shading
        vec3 normal = vec3(0.0);
        if (hitSide == 0) normal.x = -float(step.x);
        else if (hitSide == 1) normal.y = -float(step.y);
        else normal.z = -float(step.z);

        float diffuse = max(dot(normal, LIGHT_DIR), 0.2);
        
        // AO effect (fake)
        float ao = 1.0 - (float(pos.y) / 32.0) * 0.3;
        
        vec3 color = vec3(0.4, 0.7, 0.3); // Grass color
        if (pos.y < 5) color = vec3(0.4, 0.3, 0.2); // Dirt

        finalColor = vec4(color * diffuse * ao, 1.0);
    } else {
        // Sky
        float sky = 0.5 + 0.5 * rd.y;
        finalColor = vec4(vec3(0.5, 0.7, 1.0) * sky, 1.0);
    }
}
