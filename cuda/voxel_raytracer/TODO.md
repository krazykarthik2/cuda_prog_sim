Core Rendering Requirements

The system must:

Render a voxel world in real time using CUDA ray traversal.

Support free camera movement (WASD + mouse).

Maintain ≥30–60 FPS at interactive resolution.

Support a world size of at least 128³ (preferably 256³).

🌍 World Environment Requirements

The world must include:

🌄 Procedural terrain (hills or plains)

🌳 At least one voxel tree model

🌤 Sky gradient

🌞 Directional sunlight

🌫 Light atmospheric effect (optional fog)

🐦 Living Elements Requirements

At least 3 animated systems:

🐦 Birds

Small moving voxel objects

Flocking or simple circular flight

Animated wing motion OR bobbing motion

🌬 Wind System

Subtle periodic movement

Tree leaves sway

Grass sway (if implemented)

🐄 Cattle / Animals

Simple moving voxel models

Random wandering

Head bobbing or idle animation

Movement must be smooth and time-based.

💡 Lighting Requirements

Minimum:

Diffuse shading

Strong version:

Hard shadows

Ambient occlusion

Day/night cycle (slow sun movement)

🌳 Environmental Animation Requirements

At least two:

Tree leaf sway

Grass movement

Moving clouds

Water ripple

Particle effects (dust, pollen)

These should update every frame.

⚡ GPU Utilization Requirement

CUDA must handle:

Ray traversal

Lighting

World rendering

At least one dynamic system (birds or wind or animation math)

CPU should NOT be doing the rendering work.

🎮 Interactivity Requirements

The user must be able to:

Move camera

Toggle living systems

Change time of day (optional)

Regenerate terrain (optional)

🏆 Strong Version (Portfolio-Level)

Add:

Day-night cycle

Shadow length changes with sun

Reflections on water

Procedural tree placement

Simple flocking behavior for birds

Herd behavior for cattle

Now it becomes:

Real-Time CUDA Procedural Living Voxel World

That sounds serious.

🎨 What Makes It Visually Alive

Movement everywhere:

Slight grass sway

Birds circling

Cattle walking

Sun moving

Subtle fog shifting

Small motion makes huge difference.