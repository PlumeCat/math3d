#pragma once

struct perlin_noise_1d {};
struct perlin_noise_2d {
    float noise(const vec2& pos);
    float fractal_noise(const vec2& pos);
};
struct perlin_noise_3d {
    float noise(const vec3& pos);
    float fractal_noise(const vec3& pos);
};
struct perlin_noise_4d {};
struct simplex_noise_2d {
    float noise();
    float fractal_noise();
};