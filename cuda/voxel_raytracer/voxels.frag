#version 330

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 cameraUp;
uniform vec2 resolution;
uniform float time;

const int MAX_STEPS = 100;
const vec3 WATER_COLOR = vec3(0.1, 0.4, 0.8);

// Super simple hash for deterministic "randomness"
float hash11(float n) { return fract(sin(n) * 43758.5453); }

// Custom 2D Noise for Clouds
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash11(i.x + i.y * 57.0);
    float b = hash11(i.x + 1.0 + i.y * 57.0);
    float c = hash11(i.x + (i.y + 1.0) * 57.0);
    float d = hash11(i.x + 1.0 + (i.y + 1.0) * 57.0);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Voxel "Living World" Logic
// Returns: 0:Empty, 1:Grass/Dirt, 2:Stone, 3:Water, 4:Wood, 5:Leaves, 6:Animal, 7:Bird
int getVoxel(ivec3 p, vec3 worldP) {
    if (p.y < 0) return 0;
    if (p.y > 64) return 0;

    // --- LARGE VOXELS (Terrain) ---
    float h = sin(float(p.x) * 0.1) * 3.0 + cos(float(p.z) * 0.1) * 3.0 + 8.0;
    if (float(p.y) < h) {
        if (p.y > 14) return 2; // Stone
        return 1; // Grass
    }
    if (p.y < 6) return 3; // Water

    // --- SUB-VOXEL DETAIL (Animals & Trees) ---
    
    // 1. Trees (4x Smaller Voxels) - High Detail
    if (p.y < 25) { // Only check trees near ground
        float treeHash = hash11(floor(float(p.x) / 24.0) + floor(float(p.z) / 24.0) * 123.4);
        if (treeHash > 0.96) {
            ivec2 treePos = ivec2(floor(float(p.x) / 24.0) * 24 + 10, floor(float(p.z) / 24.0) * 24 + 10);
            float treeH = sin(float(treePos.x) * 0.1) * 3.0 + cos(float(treePos.y) * 0.1) * 3.0 + 8.0;
            int treeBaseY = int(floor(treeH)) + 1;
            
            if (p.x == treePos.x && p.z == treePos.y && p.y >= treeBaseY && p.y < treeBaseY + 6) {
                vec3 lp = fract(worldP * 4.0);
                if (lp.x > 0.2 && lp.x < 0.8 && lp.z > 0.2 && lp.z < 0.8) return 4;
            }
            
            float wind = sin(time * 2.0 + float(p.y) * 0.5) * 0.4;
            vec3 leafP = worldP - vec3(treePos.x, treeBaseY + 7, treePos.y);
            leafP.xz += vec2(wind); 
            if (length(leafP) < 3.5 && p.y >= treeBaseY + 4) {
                if (hash11(floor(worldP.x*4.0) + floor(worldP.y*4.0) + floor(worldP.z*4.0)) > 0.3) return 5;
            }
        }
    }

    // 2. Bird Flocking (Detail Voxel) - High Sky
    if (p.y > 28) { // Only check birds in flight
        for (int i = 0; i < 4; i++) {
            float f = float(i) * 1.5;
            float birdTime = time * 2.5 + f;
            vec3 bPos = vec3(sin(birdTime) * 20.0 + f, 35.0 + sin(birdTime * 0.4) * 6.0, cos(birdTime) * 20.0 + f*0.5);
            float wing = sin(time * 15.0 + f);
            
            vec3 dP = (worldP - bPos) * 8.0; 
            ivec3 ip = ivec3(floor(dP));
            if (ip.x == 0 && ip.z == 0 && ip.y == 0) return 7; 
            if (ip.x == -1 && ip.z == 0 && ip.y == 0) return 7; 
            if (abs(ip.x) <= 2 && ip.z == 0 && ip.y == int(wing * 2.0) && abs(ip.x) > 0) return 7; 
        }
    }

    // 3. Cattle Herd (Detail Voxel) - Near Ground
    if (p.y > 6 && p.y < 18) { // Bound to surface
        for (int i = 0; i < 3; i++) {
            float f = float(i) * 2.2;
            float cattleTime = time * 0.4 + f;
            vec2 cPos = vec2(sin(cattleTime) * 15.0 + 30.0 + f*2.0, cos(cattleTime * 0.6) * 12.0 + 30.0 + f);
            float groundH = sin(cPos.x * 0.1) * 3.0 + cos(cPos.y * 0.1) * 3.0 + 8.0;
            vec3 basePos = vec3(cPos.x, groundH + 1.2, cPos.y);
            
            vec3 dP = (worldP - basePos) * 5.0; 
            ivec3 ip = ivec3(floor(dP));
            float bob = sin(time * 4.0 + f) * 0.5;
            if (abs(ip.x) <= 1 && abs(ip.z) <= 1 && (ip.y == 0 || ip.y == 1)) return 6; 
            if (ip.x == 2 && ip.z == 0 && ip.y == int(1.0 + bob)) return 6; 
        }
    }

    // 5. Clouds (High Sky)
    if (p.y > 48) {
        float cloudTime = time * 0.5;
        float cloudH = noise(vec2(p.xz) * 0.05 + cloudTime * 0.1);
        if (p.y > 50 && p.y < 54 && cloudH > 0.7) return 8; 
    }

    return 0;
}

void main() {
    float dayCycle = time * 0.05; // Slower cycle
    vec3 sunDir = normalize(vec3(sin(dayCycle), cos(dayCycle), 0.5));
    vec3 moonDir = normalize(vec3(sin(dayCycle + 3.14), cos(dayCycle + 3.14), -0.5));
    bool isNight = sunDir.y < -0.1;

    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / resolution.y;
    vec3 right = normalize(cross(cameraDir, cameraUp));
    vec3 up = normalize(cross(right, cameraDir));
    vec3 rd = normalize(cameraDir + uv.x * right + uv.y * up);
    vec3 ro = cameraPos;

    ivec3 pos = ivec3(floor(ro));
    vec3 deltaDist = abs(1.0 / rd);
    ivec3 step = ivec3(sign(rd));
    vec3 sideDist = (sign(rd) * (vec3(pos) - ro) + (sign(rd) * 0.5) + 0.5) * deltaDist;

    int voxelType = 0;
    float dist = 0.0;
    int hitSide = 0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        voxelType = getVoxel(pos, ro + rd * (dist + 0.01));
        if (voxelType != 0) break;
        
        if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
            dist = sideDist.x;
            sideDist.x += deltaDist.x; pos.x += step.x; hitSide = 0;
        } else if (sideDist.y < sideDist.z) {
            dist = sideDist.y;
            sideDist.y += deltaDist.y; pos.y += step.y; hitSide = 1;
        } else {
            dist = sideDist.z;
            sideDist.z += deltaDist.z; pos.z += step.z; hitSide = 2;
        }
    }

    // Sky with Celestial Bodies
    vec3 daySky = mix(vec3(0.4, 0.6, 1.0), vec3(0.1, 0.3, 0.7), rd.y * 0.5 + 0.5);
    vec3 nightSky = vec3(0.01, 0.01, 0.04);
    vec3 skyColor = mix(nightSky, daySky, clamp(sunDir.y * 5.0 + 0.5, 0.0, 1.0));
    
    float sunMask = pow(max(dot(rd, sunDir), 0.0), 128.0);
    float moonMask = pow(max(dot(rd, moonDir), 0.0), 32.0);
    skyColor += vec3(1.0, 0.9, 0.5) * sunMask;
    skyColor += vec3(0.5, 0.7, 1.0) * moonMask * 0.6; // Soft blue moon

    if (voxelType != 0) {
        vec3 normal = vec3(0.0);
        if (hitSide == 0) normal.x = -float(step.x);
        else if (hitSide == 1) normal.y = -float(step.y);
        else normal.z = -float(step.z);

        vec3 color = vec3(0.5);
        if (voxelType == 1) color = vec3(0.35, 0.6, 0.2); // Grass
        else if (voxelType == 2) color = vec3(0.5, 0.5, 0.5); // Stone
        else if (voxelType == 3) color = vec3(0.1, 0.4, 0.8); // Water
        else if (voxelType == 4) color = vec3(0.4, 0.3, 0.1); // Wood
        else if (voxelType == 5) color = vec3(0.2, 0.5, 0.1); // Leaves
        else if (voxelType == 6) color = vec3(0.8, 0.8, 0.7); // Cattle
        else if (voxelType == 7) color = vec3(0.05, 0.05, 0.05); // Bird
        else if (voxelType == 8) color = vec3(0.95, 0.95, 0.98); // Cloud

        vec3 activeLight = isNight ? moonDir : sunDir;
        float diffuse = max(dot(normal, activeLight), 0.1);
        if (isNight) diffuse *= 1.5; // Bump moon influence

        vec3 finalC = color * diffuse;

        if (voxelType == 3) {
            vec3 refRd = reflect(rd, normal);
            float refAtt = pow(max(dot(refRd, activeLight), 0.0), 32.0);
            finalC = mix(finalC, skyColor + refAtt, 0.4);
        }

        float fog = exp(-dist * 0.012);
        finalColor = vec4(mix(skyColor, finalC, fog), 1.0);
    } else {
        finalColor = vec4(skyColor, 1.0);
    }
}
