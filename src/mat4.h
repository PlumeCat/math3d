#pragma once

struct mat4 {
    #ifdef JMATH_ENABLE_SSE2
    // SIMD rows[4];
    #else
    float m[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    #endif

    static mat4 identity();
    static mat4 translate(const vec3& pos);
    static mat4 scale(const vec3& scale);
    static mat4 rotate_x(float x);
    static mat4 rotate_y(float y);
    static mat4 rotate_z(float z);
    static mat4 rotate_quat(const vec4& q);
    // Right handed look-at (for camera view matrix)
    static mat4 look_at(const vec3& pos, const vec3& at, const vec3& up);
    static mat4 look(const vec3& look, const vec3& up);
    // Right handed perspective projection with clip space [ 0, 1 ] and inverted Y (for vulkan)
    static mat4 perspective(float angle, float aspect, float zn, float zf);
    static mat4 ortho(float x, float y, float w, float h, float zn, float zf);
    static mat4 world(const vec3& fd, const vec3& up, const vec3& pos);

    mat4 operator * (const mat4& _) const;
    mat4 operator * (float f) {
        return mat4 {
            m[0]/f, m[1]/f, m[2]/f, m[3]/f,
            m[4]/f, m[5]/f, m[6]/f, m[7]/f,
            m[8]/f, m[9]/f, m[10]/f, m[11]/f,
            m[12]/f, m[13]/f, m[14]/f, m[15]/f,
        };
    }

    mat4() {}
    mat4(
        float _0, float _1,  float _2,  float _3,
        float _4,  float _5,  float _6,  float _7,
        float _8,  float _9,  float _10, float _11,
        float _12, float _13, float _14, float _15
    ): m {
        _0, _1, _2, _3,
        _4, _5, _6, _7,
        _8, _9, _10,_11,
        _12,_13,_14,_15
    } {}
    mat4(const vec4& r0, const vec4& r1, const vec4& r2, const vec4& r3
    ): m {
        r0.x, r0.y, r0.z, r0.w,
        r1.x, r1.y, r1.z, r1.w,
        r2.x, r2.y, r2.z, r2.w,
        r3.x, r3.y, r3.z, r3.w,
    } {}
};