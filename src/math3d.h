#pragma once

#include <cmath>
#include <iostream>
// #ifndef JMATH_ENABLE_SSE2
// #define JMATH_ENABLE_SSE2
// #endif

// #ifdef JMATH_ENABLE_SSE2
// #include <xmmintrin.h>
// #endif

// using SIMD = __m128;

static const float pi = 3.1415926535f;
static const float degtorad = pi / 180.f;
static const float radtodeg = 180.f / pi;

struct vec2 {
    union {
        struct { float x, y; };
        float v[2];
    };

    vec2() { x = 0; y = 0; }
    vec2(float x, float y): x(x), y(y) {}
    vec2(const vec2&) = default;
    vec2(vec2&&) = default;
    vec2& operator=(const vec2&) = default;
    vec2& operator=(vec2&&) = default;

    inline vec2 operator - () const { return { -x, -y }; }

    inline vec2& operator += (const vec2& v) { *this = *this + v; return *this; }
    inline vec2& operator -= (const vec2& v) { *this = *this - v; return *this; }
    inline vec2& operator *= (const vec2& v) { *this = *this * v; return *this; }
    inline vec2& operator /= (const vec2& v) { *this = *this / v; return *this; }
    inline vec2& operator *= (const float f) { *this = *this * f; return *this; }
    inline vec2& operator /= (const float f) { *this = *this / f; return *this; }

    inline vec2 operator + (const vec2& r) const { return vec2 { x + r.x, y + r.y }; }
    inline vec2 operator - (const vec2& r) const { return vec2 { x - r.x, y - r.y }; }
    inline vec2 operator * (const vec2& r) const { return vec2 { x * r.x, y * r.y }; }
    inline vec2 operator / (const vec2& r) const { return vec2 { x / r.x, y / r.y }; }
    inline vec2 operator * (const float f) const { return vec2 { x * f, y * f }; }
    inline vec2 operator / (const float f) const { return vec2 { x / f, y / f }; }

    inline bool operator == (const vec2& r) const { return x == r.x && y == r.y; }
};

struct vec3 {
    union {
        struct { float x, y, z; };
        float v[3];
    };

    vec3() { x = 0; y = 0; z = 0; }
    vec3(float x, float y, float z): x(x), y(y), z(z) {}
    vec3(const vec3&) = default;
    vec3(vec3&&) = default;
    vec3& operator=(const vec3&) = default;
    vec3& operator=(vec3&&) = default;

    inline vec3 operator - () const { return { -x, -y, -z }; }

    inline vec3& operator += (const vec3& o) { *this = *this + o; return *this; }
    inline vec3& operator -= (const vec3& o) { *this = *this - o; return *this; }
    inline vec3& operator *= (const vec3& v) { *this = *this * v; return *this; }
    inline vec3& operator /= (const vec3& v) { *this = *this / v; return *this; }
    inline vec3& operator *= (const float f) { *this = *this * f; return *this; }
    inline vec3& operator /= (const float f) { *this = *this / f; return *this; }

    inline vec3 operator + (const vec3& r) const { return vec3 { x + r.x, y + r.y, z + r.z }; }
    inline vec3 operator - (const vec3& r) const { return vec3 { x - r.x, y - r.y, z - r.z }; }
    inline vec3 operator * (const vec3& r) const { return vec3 { x * r.x, y * r.y, z * r.z }; }
    inline vec3 operator / (const vec3& r) const { return vec3 { x / r.x, y / r.y, z / r.z }; }
    inline vec3 operator * (const float f) const { return vec3 { x * f, y * f, z * f }; }
    inline vec3 operator / (const float f) const { return vec3 { x / f, y / f, z / f }; }

    inline bool operator == (const vec3& r) const { return x == r.x && y == r.y && z == r.z; }

};

struct vec4 {
    union {
        struct { float x, y, z, w; };
        float v[4];
    };

    vec4() { x = 0; y = 0; z = 0; w = 0; }
    vec4(float x, float y, float z, float w): x(x), y(y), z(z), w(w) {}
    vec4(const vec4&) = default;
    vec4(vec4&&) = default;
    vec4& operator=(const vec4&) = default;
    vec4& operator=(vec4&&) = default;

    inline vec4 operator - () const { return { -x, -y, -z, -w }; }

    inline vec4& operator += (const vec4& o) { *this = *this + o; return *this; }
    inline vec4& operator -= (const vec4& o) { *this = *this - o; return *this; }
    inline vec4& operator *= (const vec4& v) { *this = *this * v; return *this; }
    inline vec4& operator /= (const vec4& v) { *this = *this / v; return *this; }
    inline vec4& operator *= (const float f) { *this = *this * f; return *this; }
    inline vec4& operator /= (const float f) { *this = *this / f; return *this; }

    inline vec4 operator + (const vec4& r) const { return vec4 { x + r.x, y + r.y, z + r.z, w + r.w }; }
    inline vec4 operator - (const vec4& r) const { return vec4 { x - r.x, y - r.y, z - r.z, w - r.w }; }
    inline vec4 operator * (const vec4& r) const { return vec4 { x * r.x, y * r.y, z * r.z, w * r.w }; }
    inline vec4 operator / (const vec4& r) const { return vec4 { x / r.x, y / r.y, z / r.z, w / r.w }; }
    inline vec4 operator * (const float f) const { return vec4 { x * f, y * f, z * f, w * f }; }
    inline vec4 operator / (const float f) const { return vec4 { x / f, y / f, z / f, w / f }; }

    inline bool operator == (const vec4& r) const { return x == r.x && y == r.y && z == r.z && w == r.w; }
};

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
    // Right handed look-at (for camera view matrix)
    static mat4 look_at(const vec3& pos, const vec3& at, const vec3& up);
    // Right handed perspective projection with clip space [ 0, 1 ] and inverted Y (for vulkan)
    static mat4 perspective(float angle, float aspect, float zn, float zf);
    static mat4 ortho(float w, float h, float zn, float zf);
    static mat4 world(const vec3& fd, const vec3& up, const vec3& pos);

    mat4 operator * (const mat4& _);
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

float dot(const vec2& u, const vec2& v);
float dot(const vec3& u, const vec3& v);
float dot(const vec4& u, const vec4& v);

vec3 cross(const vec3& u, const vec3& v);
vec3 reflect(const vec3& v, const vec3& n);
vec3 refract(const vec3& v, const vec3& n, float i);

vec2 normalize(const vec2& v);
vec3 normalize(const vec3& v);
vec4 normalize(const vec4& v);

float length(const vec2& v);
float length(const vec3& v);
float length(const vec4& v);

float length_sq(const vec2& v);
float length_sq(const vec3& v);
float length_sq(const vec4& v);

float distance(const vec2& a, const vec2& b);
float distance(const vec3& a, const vec3& b);
float distance(const vec4& a, const vec4& b);

vec2 lerp(const vec2& x, const vec2& y, const float t);
vec3 lerp(const vec3& x, const vec3& y, const float t);
vec4 lerp(const vec4& x, const vec4& y, const float t);

vec2 mul(const vec2& v, const mat4& m);
vec3 mul(const vec3& v, const mat4& m);
vec4 mul(const vec4& v, const mat4& m);
vec2 mul_norm(const vec2& v, const mat4& m);
vec3 mul_norm(const vec3& v, const mat4& m);

mat4 transpose(const mat4& m);
mat4 inverse(const mat4& m);
float determinant(const mat4& m);
float degrees(float r);
float radians(float d);
float saturate(float x);

#ifdef JMATH_IMPLEMENTATION
#define per_component_2(f) vec2 f(const vec2& v) { return { f(v.x), f(v.y) }; }
#define per_component_3(f) vec3 f(const vec3& v) { return { f(v.x), f(v.y), f(v.z) }; }
#define per_component_4(f) vec4 f(const vec4& v) { return { f(v.x), f(v.y), f(v.z), f(v.w) }; }
#define per_component_2_2(f) vec2 f(const vec2& v, float s) { return { f(v.x, s), f(v.y, s) }; }
#define per_component_2_3(f) vec3 f(const vec3& v, float s) { return { f(v.x, s), f(v.y, s), f(v.z, s) }; }
#define per_component_2_4(f) vec4 f(const vec4& v, float s) { return { f(v.x, s), f(v.y, s), f(v.z, s), f(v.w, s) }; }
#else
#define per_component_2(f) vec2 f(const vec2& v);
#define per_component_3(f) vec3 f(const vec3& v);
#define per_component_4(f) vec4 f(const vec4& v);
#define per_component_2_2(f) vec2 f(const vec2& v, float s);
#define per_component_2_3(f) vec3 f(const vec3& v, float s);
#define per_component_2_4(f) vec4 f(const vec4& v, float s);
#endif
#define per_component_all(func)\
    per_component_2(func)\
    per_component_3(func)\
    per_component_4(func)
#define per_component_all_2(func)\
    per_component_2_2(func)\
    per_component_2_3(func)\
    per_component_2_4(func)

per_component_all(sin)
per_component_all(cos)
per_component_all(tan)
per_component_all(asin)
per_component_all(acos)
per_component_all(atan)
per_component_all(sinh)
per_component_all(cosh)
per_component_all(tanh)
per_component_all(asinh)
per_component_all(acosh)
per_component_all(atanh)

per_component_all(exp)
per_component_all(exp2)
per_component_all(sqrt)
per_component_all(log)
per_component_all(log2)
per_component_all(log10)

per_component_all(floor)
per_component_all(ceil)
per_component_all(abs)
per_component_all(round)
per_component_all(degrees)
per_component_all(radians)

per_component_all(trunc)

// fmod
// frexp
// ldexp
// mod
// frac
// modf
// pow
// sign
// smoothstep
// smootherstep
// lerp
// slerp
// clamp
// atan2 in some form
// min
// max

/*

matrix
    frustums
    perspective (infinite, lh, rh, fov, ...), ortho (lh, rh)
    look_at
    pick_matrix (??)
    project (??) unproject (??)
    rotate_x, rotate_y, rotate_z, rotate_axis, rotate_euler
    scale
    translate

color
    linear to srgb etc

packing
    byte
    byten
    ubyten
    short
    shortn
    ushortn
    half

some intersection stuff?
    closest_point(line, point)
    line_sphere
    line_box
    line_triangle
    line_plane

*/



#ifdef JMATH_IMPLEMENTATION

float degrees(float r) {
    return r * radtodeg;
}
float radians(float d) {
    return d * degtorad;
}
float saturate(float x) {
    return std::fmin(std::fmax(x, 0.f), 1.f);
}

mat4 mat4::identity() {
    return mat4 {};
}

mat4 mat4::translate(const vec3& pos) {
    return mat4 {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos.x, pos.y, pos.z, 1
    };
}
mat4 mat4::scale(const vec3& scale) {
    return mat4 {
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, scale.z, 0,
        0, 0, 0, 1
    };
}
mat4 mat4::rotate_x(float x) {
    float c = cos(x);
    float s = sin(x);
    return mat4 {
        1, 0, 0, 0,
        0, c, -s, 0,
        0, s, c, 0,
        0, 0, 0, 1
    };
}
mat4 mat4::rotate_y(float y) {
    float c = cos(y);
    float s = sin(y);
    return mat4 {
        c, 0, -s, 0,
        0, 1, 0,  0,
        s, 0, c,  0,
        0, 0, 0,  1
    };
}
mat4 mat4::rotate_z(float z) {
    float c = cos(z);
    float s = sin(z);
    return mat4 {
        c, -s, 0, 0,
        s,  c, 0, 0,
        0,  0, 1, 0,
        0,  0, 0, 1
    };
}
mat4 mat4::look_at(const vec3& pos, const vec3& at, const vec3& up) {
    auto az = normalize(at - pos);
    auto ax = normalize(cross(az, up));
    auto ay = cross(ax, az);

    return mat4 {
        ax.x,		   ay.x,		  -az.x,	     0,
        ax.y,		   ay.y,		  -az.y,	     0,
        ax.z,		   ay.z,		  -az.z,	     0,
        -dot(ax, pos), -dot(ay, pos), dot(az, pos),  1
    };


    //template<typename T, qualifier Q>
    //GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookAtRH(vec<3, T, Q> const& eye, vec<3, T, Q> const& center, vec<3, T, Q> const& up) {
    //	vec<3, T, Q> const f(normalize(center - eye));
    //	vec<3, T, Q> const s(normalize(cross(f, up)));
    //	vec<3, T, Q> const u(cross(s, f));

    //	mat<4, 4, T, Q> Result(1);
    //	Result[0] = s.x;
    //	Result[4] = s.y;
    //	Result[8] = s.z;
    //	Result[1] = u.x;
    //	Result[5] = u.y;
    //	Result[9] = u.z;
//	Result[2] = -f.x;
    //	Result[6] = -f.y;
    //	Result[10] = -f.z;
    //	Result[12] = -dot(s, eye);
    //	Result[13] = -dot(u, eye);
    //	Result[14] = dot(f, eye);
    //	return Result;
    //}
}
mat4 mat4::perspective(float angle, float aspect, float zn, float zf) {
    auto ys = 1.f / tan(angle / 2.f);
    auto xs = ys / aspect;
    auto d = zn - zf;
    auto zs = zf / d;
    auto zb = zn * zf / d;

    return mat4 {
        xs, 0, 0, 0,
        0, -ys, 0, 0,
        0, 0, zs, -1,
        0, 0, zb, 0
    };
    /*
    * Adapted from GLM source
    
    mat4 perspectiveFovRH_ZO(float fov, float width, float height, float zNear, float zFar) {
        T const rad = fov;
        T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
        T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

        mat<4, 4, T, defaultp> Result(static_cast<T>(0));
        Result[0] = w;
        Result[5] = h;
        Result[10] = zFar / (zNear - zFar);
        Result[11] = - static_cast<T>(1);
        Result[14] = -(zFar * zNear) / (zFar - zNear);
        return Result;
    }

    // rewritten:
    {
        float rad = fov;
        float y = cos(0.5 * rad) / sin(0.5 * rad)
                = 1 / tan(0.5 * rad)

        float x = y / aspect
        float z = zfar / (znear - zfar)
        float w = -(zfar * znear) / (zfar - znear)

        return {
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, w,
            0, 0, -1, 0
        }
    }
    */
}
mat4 mat4::ortho(float w, float h, float zn, float zf) {
    auto zd = zn - zf;
    return mat4 {
        2 / (w), 0, 0, 0,
        0, 2 / h, 0, 0,
        0, 0, 1 / zd, 0,
        0, 0, zn / zd, 1,
    };
    /*
    Adapted from GLM source
        template<typename T>
        GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_ZO(T left, T right, T bottom, T top, T zNear, T zFar)
        {
            mat<4, 4, T, defaultp> Result(1);
            Result[0] = static_cast<T>(2) / (right - left);
            Result[5] = static_cast<T>(2) / (top - bottom);
            Result[10] = - static_cast<T>(1) / (zFar - zNear);
            Result[12] = - (right + left) / (right - left);
            Result[13] = - (top + bottom) / (top - bottom);
            Result[14] = - zNear / (zFar - zNear);
            return Result;
        }
    */
}

mat4 mat4::world(const vec3& fd, const vec3& up, const vec3& pos) {
    const auto lt = cross(up, fd);
    return mat4 {
        lt.x,  up.x,  fd.x,  0,
        lt.y,  up.y,  fd.y,  0,
        lt.z,  up.z,  fd.z,  0,
        pos.x, pos.y, pos.z, 1
    };
}

mat4 mat4::operator*(const mat4& _) {
    // TODO: vectorize
    return mat4 {
        m[0] * _.m[0] + m[1] * _.m[4] + m[2] * _.m[8] + m[3] * _.m[12],
        m[0] * _.m[1] + m[1] * _.m[5] + m[2] * _.m[9] + m[3] * _.m[13],
        m[0] * _.m[2] + m[1] * _.m[6] + m[2] * _.m[10] + m[3] * _.m[14],
        m[0] * _.m[3] + m[1] * _.m[7] + m[2] * _.m[11] + m[3] * _.m[15],

        m[4] * _.m[0] + m[5] * _.m[4] + m[6] * _.m[8] + m[7] * _.m[12],
        m[4] * _.m[1] + m[5] * _.m[5] + m[6] * _.m[9] + m[7] * _.m[13],
        m[4] * _.m[2] + m[5] * _.m[6] + m[6] * _.m[10] + m[7] * _.m[14],
        m[4] * _.m[3] + m[5] * _.m[7] + m[6] * _.m[11] + m[7] * _.m[15],

        m[8] * _.m[0] + m[9] * _.m[4] + m[10] * _.m[8] + m[11] * _.m[12],
        m[8] * _.m[1] + m[9] * _.m[5] + m[10] * _.m[9] + m[11] * _.m[13],
        m[8] * _.m[2] + m[9] * _.m[6] + m[10] * _.m[10] + m[11] * _.m[14],
        m[8] * _.m[3] + m[9] * _.m[7] + m[10] * _.m[11] + m[11] * _.m[15],

        m[12] * _.m[0] + m[13] * _.m[4] + m[14] * _.m[8] + m[15] * _.m[12],
        m[12] * _.m[1] + m[13] * _.m[5] + m[14] * _.m[9] + m[15] * _.m[13],
        m[12] * _.m[2] + m[13] * _.m[6] + m[14] * _.m[10] + m[15] * _.m[14],
        m[12] * _.m[3] + m[13] * _.m[7] + m[14] * _.m[11] + m[15] * _.m[15]
    };
}

float dot(const vec2& u, const vec2& v) {
    return u.x * v.x + u.y * v.y;
}
float dot(const vec3& u, const vec3& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
float dot(const vec4& u, const vec4& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

vec3 cross(const vec3& u, const vec3& v) {
    return {
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    };
}

vec2 normalize(const vec2& v) {
    return v / length(v);
}
vec3 normalize(const vec3& v) {
    return v / length(v);
}
vec4 normalize(const vec4& v) {
    return v / length(v);
}

float length(const vec2& v) {
    return sqrt(dot(v, v));
}
float length(const vec3& v) {
    return sqrt(dot(v, v));
}
float length(const vec4& v) {
    return sqrt(dot(v, v));
}

float length_sq(const vec2& v) {
    return (dot(v, v));
}
float length_sq(const vec3& v) {
    return (dot(v, v));
}
float length_sq(const vec4& v) {
    return (dot(v, v));
}

vec2 lerp(const vec2& x, const vec2& y, const float t) {
    return x + (y - x) * t;
}
vec3 lerp(const vec3& x, const vec3& y, const float t) {
    return x + (y - x) * t;
}
vec4 lerp(const vec4& x, const vec4& y, const float t) {
    return x + (y - x) * t;
}

// template<int r1, int r2, int c1, int c2>
// float determinant2x2(const mat4& m) {
//     return m.m[r1][c1] * m.m[r2][c2] - m.m[r2][c1] * m.m[r1][c2];
// }
// template<int r1, int r2, int r3, int c1, int c2, int c3>
// float determinant3x3(const mat4& m) {
//     return
//         m.m[r1][c1] * determinant2x2<r2, r3, c2, c3>(m) +
//         m.m[r1][c2] * -determinant2x2<r2, r3, c1, c3>(m) +
//         m.m[r1][c3] * determinant2x2<r2, r3, c1, c2>(m);
// }
// float determinant(const mat4& m) {
//     return
//         m.m[0] * determinant3x3<1, 2, 3, 1, 2, 3>(m) +
//         m.m[1] * -determinant3x3<1, 2, 3, 0, 2, 3>(m) +
//         m.m[2] * determinant3x3<1, 2, 3, 0, 1, 3>(m) +
//         m.m[3] * -determinant3x3<1, 2, 3, 0, 1, 2>(m);
// }



mat4 inverse(const mat4& m) {
    // T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    // T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    // T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

    // T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    // T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    // T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

    // T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    // T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
    // T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

    // T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    // T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
    // T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

    // T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    // T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    // T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

    // vec<4, T, Q> Fac0(Coef00, Coef00, Coef02, Coef03);
    // vec<4, T, Q> Fac1(Coef04, Coef04, Coef06, Coef07);
    // vec<4, T, Q> Fac2(Coef08, Coef08, Coef10, Coef11);
    // vec<4, T, Q> Fac3(Coef12, Coef12, Coef14, Coef15);
    // vec<4, T, Q> Fac4(Coef16, Coef16, Coef18, Coef19);
    // vec<4, T, Q> Fac5(Coef20, Coef20, Coef22, Coef23);

    // vec<4, T, Q> Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
    // vec<4, T, Q> Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
    // vec<4, T, Q> Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
    // vec<4, T, Q> Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

    // vec<4, T, Q> Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
    // vec<4, T, Q> Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
    // vec<4, T, Q> Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
    // vec<4, T, Q> Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

    // vec<4, T, Q> SignA(+1, -1, +1, -1);
    // vec<4, T, Q> SignB(-1, +1, -1, +1);
    // mat<4, 4, T, Q> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

    // vec<4, T, Q> Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

    // vec<4, T, Q> Dot0(m[0] * Row0);
    // T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

    // T OneOverDeterminant = static_cast<T>(1) / Dot1;

    // return Inverse * OneOverDeterminant;
    float coef00 = m.m[10] * m.m[15] - m.m[14] * m.m[11];
    float coef02 = m.m[6] * m.m[15] - m.m[14] * m.m[7];
    float coef03 = m.m[6] * m.m[11] - m.m[10] * m.m[7];

    float coef04 = m.m[9] * m.m[15] - m.m[13] * m.m[11];
    float coef06 = m.m[5] * m.m[15] - m.m[13] * m.m[7];
    float coef07 = m.m[5] * m.m[11] - m.m[9] * m.m[7];

    float coef08 = m.m[9] * m.m[14] - m.m[13] * m.m[10];
    float coef10 = m.m[5] * m.m[14] - m.m[13] * m.m[6];
    float coef11 = m.m[5] * m.m[10] - m.m[9] * m.m[6];

    float coef12 = m.m[8] * m.m[15] - m.m[12] * m.m[11];
    float coef14 = m.m[4] * m.m[15] - m.m[12] * m.m[7];
    float coef15 = m.m[4] * m.m[11] - m.m[8] * m.m[7];

    float coef16 = m.m[8] * m.m[14] - m.m[12] * m.m[10];
    float coef18 = m.m[4] * m.m[14] - m.m[12] * m.m[6];
    float coef19 = m.m[4] * m.m[10] - m.m[8] * m.m[6];

    float coef20 = m.m[8] * m.m[13] - m.m[12] * m.m[9];
    float coef22 = m.m[4] * m.m[13] - m.m[12] * m.m[5];
    float coef23 = m.m[4] * m.m[9] -  m.m[8] *  m.m[5];

    auto fac0 = vec4(coef00, coef00, coef02, coef03);
    auto fac1 = vec4(coef04, coef04, coef06, coef07);
    auto fac2 = vec4(coef08, coef08, coef10, coef11);
    auto fac3 = vec4(coef12, coef12, coef14, coef15);
    auto fac4 = vec4(coef16, coef16, coef18, coef19);
    auto fac5 = vec4(coef20, coef20, coef22, coef23);

    auto vec0 = vec4(m.m[4], m.m[0], m.m[0], m.m[0]);
    auto vec1 = vec4(m.m[5], m.m[1], m.m[1], m.m[1]);
    auto vec2 = vec4(m.m[6], m.m[2], m.m[2], m.m[2]);
    auto vec3 = vec4(m.m[7], m.m[3], m.m[3], m.m[3]);

    auto inv0 = vec4(vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
    auto inv1 = vec4(vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
    auto inv2 = vec4(vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
    auto inv3 = vec4(vec0 * fac2 - vec1 * fac4 + vec2 * fac5);

    auto signa = vec4(+1, -1, +1, -1);
    auto signb = vec4(-1, +1, -1, +1);
    auto inverse = mat4(
        inv0 * signa,
        inv1 * signb,
        inv2 * signa,
        inv3 * signb
    );

    auto row0 = vec4(inverse.m[0], inverse.m[4], inverse.m[8], inverse.m[12]);
    auto dot0 = vec4(m.m[0], m.m[1], m.m[2], m.m[3]) * row0;
    float dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    float oneoverdeterminant = dot1;

    return inverse * oneoverdeterminant;
}

vec2 mul(const vec2& v, const mat4& m) {
    return vec2 {
        v.x * m.m[0]  + v.y * m.m[1]  + m.m[2]  + m.m[3],
        v.x * m.m[4]  + v.y * m.m[5]  + m.m[6]  + m.m[7]
    };
}
vec3 mul(const vec3& v, const mat4& m) {
    return vec3 {
        v.x * m.m[0]  + v.y * m.m[1]  + v.z * m.m[2]  + m.m[3],
        v.x * m.m[4]  + v.y * m.m[5]  + v.z * m.m[6]  + m.m[7],
        v.x * m.m[8]  + v.y * m.m[9]  + v.z * m.m[10] + m.m[11]
    };
}
vec4 mul(const vec4& v, const mat4& m) {
    return vec4 {
        v.x * m.m[0]  + v.y * m.m[1]  + v.z * m.m[2]  + v.w * m.m[3],
        v.x * m.m[4]  + v.y * m.m[5]  + v.z * m.m[6]  + v.w * m.m[7],
        v.x * m.m[8]  + v.y * m.m[9]  + v.z * m.m[10] + v.w * m.m[11],
        v.x * m.m[12] + v.y * m.m[13] + v.z * m.m[14] + v.w * m.m[15],
    };
}
vec2 mul_norm(const vec2& v, const mat4& m) {
    return vec2 {
        v.x * m.m[0]  + v.y * m.m[1],
        v.x * m.m[4]  + v.y * m.m[5]
    };
}
vec3 mul_norm(const vec3& v, const mat4& m) {
    return vec3 {
        v.x * m.m[0]  + v.y * m.m[1]  + v.z * m.m[2],
        v.x * m.m[4]  + v.y * m.m[5]  + v.z * m.m[6],
        v.x * m.m[8]  + v.y * m.m[9]  + v.z * m.m[10]
    };
}

std::ostream& operator << (std::ostream& o, const vec2& v) {
    return o<<"vec2 { "<<v.x<<", "<<v.y<<"}";
}
std::ostream& operator<<(std::ostream& o, const vec3& v) {
    return o<<"vec2 { "<<v.x<<", "<<v.y<<", "<<v.z<<"}";
}
std::ostream& operator<<(std::ostream& o, const vec4& v) {
    return o<<"vec2 { "<<v.x<<", "<<v.y<<", "<<v.z<<", "<<v.w<<"}";
}
std::ostream& operator<<(std::ostream& o, const mat4& m) {
    return o<<"mat4 { "<<"\n"
        <<"    "<<m.m[0] <<','<<m.m[1] <<','<<m.m[2] <<','<<m.m[3] << "\n"
        <<"    "<<m.m[4] <<','<<m.m[5] <<','<<m.m[6] <<','<<m.m[7] << "\n"
        <<"    "<<m.m[8] <<','<<m.m[9] <<','<<m.m[10]<<','<<m.m[11] << "\n"
        <<"    "<<m.m[12]<<','<<m.m[13]<<','<<m.m[14]<<','<<m.m[15] << "\n"
        << "}";
}

#endif