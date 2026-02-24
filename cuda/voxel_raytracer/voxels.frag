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
const vec3 WATER_COLOR = vec3(0.1, 0.4, 0.8);

// SUPER FAST Terrain Function (No Hashes/Noise)
int getVoxel(ivec3 p) {
    if (p.y < 0) return 1;
    if (p.y > 48) return 0;
    
    // Use simple sine waves instead of expensive noise
    float h = sin(float(p.x) * 0.1) * 4.0 + cos(float(p.z) * 0.1) * 4.0 + 10.0;
    
    if (float(p.y) < h) {
        if (p.y > 14) return 2; // Stone
        return 1; // Grass
    }
    if (p.y < 8) return 3; // Water
    return 0;
}

// FAST Shadow Trace (Lower Step Count)
float getShadow(vec3 ro, vec3 rd) {
    ivec3 pos = ivec3(floor(ro));
    vec3 deltaDist = abs(1.0 / rd);
    ivec3 step = ivec3(sign(rd));
    vec3 sideDist = (sign(rd) * (vec3(pos) - ro) + (sign(rd) * 0.5) + 0.5) * deltaDist;

    for (int i = 0; i < 20; i++) { // Very few steps
        if (getVoxel(pos) != 0) return 0.5;
        if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
            sideDist.x += deltaDist.x;
            pos.x += step.x;
        } else if (sideDist.y < sideDist.z) {
            sideDist.y += deltaDist.y;
            pos.y += step.y;
        } else {
            sideDist.z += deltaDist.z;
            pos.z += step.z;
        }
    }
    return 1.0;
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / resolution.y;
    vec3 right = normalize(cross(cameraDir, cameraUp));
    vec3 up = normalize(cross(right, cameraDir));
    vec3 rd = normalize(cameraDir + uv.x * right + uv.y * up);
    vec3 ro = cameraPos;

    ivec3 pos = ivec3(floor(ro));
    vec3 deltaDist = abs(1.0 / rd);
    ivec3 step = ivec3(sign(rd));
    vec3 sideDist = (sign(rd) * (vec3(pos) - ro) + (sign(rd) * 0.5) + 0.5) * deltaDist;

    int hitSide = 0;
    int voxelType = 0;
    float dist = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        voxelType = getVoxel(pos);
        if (voxelType != 0) break;
        
        if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
            dist = sideDist.x;
            sideDist.x += deltaDist.x;
            pos.x += step.x;
            hitSide = 0;
        } else if (sideDist.y < sideDist.z) {
            dist = sideDist.y;
            sideDist.y += deltaDist.y;
            pos.y += step.y;
            hitSide = 1;
        } else {
            dist = sideDist.z;
            sideDist.z += deltaDist.z;
            pos.z += step.z;
            hitSide = 2;
        }
    }

    vec3 skyColor = mix(vec3(0.4, 0.6, 1.0), vec3(0.1, 0.3, 0.7), rd.y * 0.5 + 0.5);
    float sun = pow(max(dot(rd, LIGHT_DIR), 0.0), 32.0);
    skyColor += vec3(1.0, 0.9, 0.5) * sun;

    if (voxelType != 0) {
        vec3 normal = vec3(0.0);
        if (hitSide == 0) normal.x = -float(step.x);
        else if (hitSide == 1) normal.y = -float(step.y);
        else normal.z = -float(step.z);

        vec3 color = vec3(1.0);
        if (voxelType == 1) color = vec3(0.4, 0.7, 0.2); // Grass
        else if (voxelType == 2) color = vec3(0.5, 0.5, 0.5); // Stone
        else if (voxelType == 3) color = WATER_COLOR; // Water

        // Shadows
        vec3 hitPos = ro + rd * dist;
        float shadow = getShadow(hitPos + normal * 0.01, LIGHT_DIR);
        
        float diffuse = max(dot(normal, LIGHT_DIR), 0.0) * shadow + 0.2;
        
        // Fake AO
        float ao = 1.0;
        if (hitSide == 1 && step.y == -1) { // Top face
            if (getVoxel(pos + ivec3(1, 0, 0)) != 0) ao *= 0.8;
            if (getVoxel(pos + ivec3(-1, 0, 0)) != 0) ao *= 0.8;
            if (getVoxel(pos + ivec3(0, 0, 1)) != 0) ao *= 0.8;
            if (getVoxel(pos + ivec3(0, 0, -1)) != 0) ao *= 0.8;
        }

        vec3 finalC = color * diffuse * ao;
        float fog = exp(-dist * 0.01);
        finalColor = vec4(mix(skyColor, finalC, fog), 1.0);
    } else {
        finalColor = vec4(skyColor, 1.0);
    }
}
