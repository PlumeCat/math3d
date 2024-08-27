#pragma once

#include <cmath>
#include <iostream>
#include <type_traits>

template<typename T> concept arithmetic = std::integral<T> || std::floating_point<T>;
template<typename T> concept signed_arithmetic = arithmetic<T> && (std::is_signed_v<T> || std::floating_point<T>);

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
static const float epsilon_f = 0.000001f;
static const float epsilon_d = 0.0000000000000001;
float degrees(float r);
float radians(float d);
float saturate(float x);
float clamp(float a, float b, float x);
float step(float x, float edge);

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat4.h"
#include "perlin_noise.h"


template<typename A, typename B> float dot(jm::vec2<A> u, jm::vec2<B> v) {
    return u.x * v.x + u.y * v.y;
}
template<typename A, typename B> float dot(const jm::vec3<A>& u, const jm::vec3<B>& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
template<typename A, typename B> float dot(const jm::vec4<A>& u, const jm::vec4<B>& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

vec3 cross(const vec3& u, const vec3& v);
vec3 reflect(const vec3& v, const vec3& n);
vec3 refract(const vec3& v, const vec3& n, float i);

vec2 sign(const vec2& v);
vec3 sign(const vec3& v);
vec4 sign(const vec4& v);

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
vec4 slerp(const vec4& x, const vec4& y, const float t);

vec2 min(const vec2& x, const vec2& y);
vec3 min(const vec3& x, const vec3& y);
vec4 min(const vec4& x, const vec4& y);
vec2 max(const vec2& x, const vec2& y);
vec3 max(const vec3& x, const vec3& y);
vec4 max(const vec4& x, const vec4& y);

vec2 mul(const vec2& v, const mat4& m);
vec3 mul(const vec3& v, const mat4& m);
vec4 mul(const vec4& v, const mat4& m);
vec2 mul_norm(const vec2& v, const mat4& m);
vec3 mul_norm(const vec3& v, const mat4& m);

mat4 transpose(const mat4& m);
mat4 inverse(const mat4& m);
float determinant(const mat4& m);

template<arithmetic A> A min(A l, A r) { return std::min(l, r); }
template<arithmetic A> A max(A l, A r) { return std::min(l, r); }

#ifdef JMATH_IMPLEMENTATION
#define per_component_v2(f) vec2 f(const vec2& v) { return { f(v.x), f(v.y) }; }
#define per_component_v3(f) vec3 f(const vec3& v) { return { f(v.x), f(v.y), f(v.z) }; }
#define per_component_v4(f) vec4 f(const vec4& v) { return { f(v.x), f(v.y), f(v.z), f(v.w) }; }
#define per_component_v2s(f) vec2 f(const vec2& v, float s) { return { f(v.x, s), f(v.y, s) }; }
#define per_component_v3s(f) vec3 f(const vec3& v, float s) { return { f(v.x, s), f(v.y, s), f(v.z, s) }; }
#define per_component_v4s(f) vec4 f(const vec4& v, float s) { return { f(v.x, s), f(v.y, s), f(v.z, s), f(v.w, s) }; }
#define per_component_v2v2(f) vec2 f(const vec2& v1, const vec2& v2) { return { f(v1.x, v2.x), f(v1.y, v2.y) }; }
#define per_component_v3v3(f) vec3 f(const vec3& v1, const vec3& v2) { return { f(v1.x, v2.x), f(v1.y, v2.y), f(v1.z, v2.z) }; }
#define per_component_v4v4(f) vec4 f(const vec4& v1, const vec4& v2) { return { f(v1.x, v2.x), f(v1.y, v2.y), f(v1.z, v2.z), f(v1.w, v2.w) }; }
#define per_component_v2v2v2(f) vec2 f(const vec2& v1, const vec2& v2, const vec2& v3) { return { f(v1.x, v2.x, v3.x), f(v1.y, v2.y, v3.y) }; }
#define per_component_v3v3v3(f) vec3 f(const vec3& v1, const vec3& v2, const vec3& v3) { return { f(v1.x, v2.x, v3.x), f(v1.y, v2.y, v3.y), f(v1.z, v2.z, v3.z) }; }
#define per_component_v4v4v4(f) vec4 f(const vec4& v1, const vec4& v2, const vec4& v3) { return { f(v1.x, v2.x, v3.x), f(v1.y, v2.y, v3.y), f(v1.z, v2.z, v3.z), f(v1.w, v2.w, v3.w) }; }
#else
#define per_component_v2(f) vec2 f(const vec2& v);
#define per_component_v3(f) vec3 f(const vec3& v);
#define per_component_v4(f) vec4 f(const vec4& v);
#define per_component_v2s(f) vec2 f(const vec2& v, float s);
#define per_component_v3s(f) vec3 f(const vec3& v, float s);
#define per_component_v4s(f) vec4 f(const vec4& v, float s);
#define per_component_v2v2(f) vec2 f(const vec2& v1, const vec2& v2);
#define per_component_v3v3(f) vec3 f(const vec3& v1, const vec3& v2);
#define per_component_v4v4(f) vec4 f(const vec4& v1, const vec4& v2);
#define per_component_v2v2v2(f) vec2 f(const vec2& v1, const vec2& v2, const vec2& v3);
#define per_component_v3v3v3(f) vec3 f(const vec3& v1, const vec3& v2, const vec3& v3);
#define per_component_v4v4v4(f) vec4 f(const vec4& v1, const vec4& v2, const vec4& v3);
#endif

#define per_component(func)\
    per_component_v2(func)\
    per_component_v3(func)\
    per_component_v4(func)
#define per_component_vs(func)\
    per_component_v2s(func)\
    per_component_v3s(func)\
    per_component_v4s(func)
#define per_component_vv(func)\
    per_component_v2v2(func)\
    per_component_v3v3(func)\
    per_component_v4v4(func)
#define per_component_vvv(func)\
    per_component_v2v2v2(func)\
    per_component_v3v3v3(func)\
    per_component_v4v4v4(func)
#define per_component_vs_vv(func)\
    per_component_vs(func)\
    per_component_vv(func)

per_component(sin)
per_component(cos)
per_component(tan)
per_component(asin)
per_component(acos)
per_component(atan)
per_component(sinh)
per_component(cosh)
per_component(tanh)
per_component(asinh)
per_component(acosh)
per_component(atanh)

per_component(exp)
per_component(exp2)
per_component(sqrt)

per_component(logf)
per_component(log2)
per_component(log10)

per_component(floor)
per_component(ceil)
per_component(abs)
per_component(round)
per_component(degrees)
per_component(radians)

per_component(trunc)

per_component_vv(step)

per_component_vs_vv(fmod)
per_component_vs_vv(pow)

per_component_vvv(clamp)

// fmod(x, y)   -> z | x = ny + z for some n
// frexp(x, &i) -> f | x = 2**i + f
// ldexp(x, i)  -> f | f = 2**i + x // inverse of frexp
// frac
// modf(x, &i)  -> f | x = i + f (i integer, 0 <= f < 1)

// pow
// sign
// smoothstep
// smootherstep
// lerp
// slerp
// clamp
// atan2 in some form ("angle" for vec2?)
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
    linear srgb etc

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
float clamp(float a, float b, float x) {
    return std::fmin(std::fmax(x, a), b);
}
float step(float edge, float x) {
    return (x < edge) ? 0.0 : 1.0;
}


vec2 sign(const vec2& v) {
    return {
        std::signbit(v.x) ? -1.f : 1.f,
        std::signbit(v.y) ? -1.f : 1.f
    };
}
vec3 sign(const vec3& v) {
    return {
        std::signbit(v.x) ? -1.f : 1.f,
        std::signbit(v.y) ? -1.f : 1.f,
        std::signbit(v.z) ? -1.f : 1.f
    };
}
vec4 sign(const vec4& v) {
    return {
        std::signbit(v.x) ? -1.f : 1.f,
        std::signbit(v.y) ? -1.f : 1.f,
        std::signbit(v.z) ? -1.f : 1.f,
        std::signbit(v.w) ? -1.f : 1.f
    };
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
mat4 mat4::rotate_quat(const vec4& q) {
    const auto q0 = q.w;
    const auto q1 = q.x;
    const auto q2 = q.y;
    const auto q3 = q.z;

    // First row of the rotation matrix
    const auto r00 = 2 * (q0 * q0 + q1 * q1) - 1;
    const auto r01 = 2 * (q1 * q2 - q0 * q3);
    const auto r02 = 2 * (q1 * q3 + q0 * q2);

    // Second row of the rotation matrix
    const auto r10 = 2 * (q1 * q2 + q0 * q3);
    const auto r11 = 2 * (q0 * q0 + q2 * q2) - 1;
    const auto r12 = 2 * (q2 * q3 - q0 * q1);

    // Third row of the rotation matrix
    const auto r20 = 2 * (q1 * q3 - q0 * q2);
    const auto r21 = 2 * (q2 * q3 + q0 * q1);
    const auto r22 = 2 * (q0 * q0 + q3 * q3) - 1;

    // 3x3 rotation matrix
    /*rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]] )*/

    return mat4 {
        r00, r01, r02, 0,
        r10, r11, r12, 0,
        r20, r21, r22, 0,
        0,   0,   0,   1
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
}
mat4 mat4::ortho(float x, float y, float w, float h, float zn, float zf) {
    auto zd = zn - zf;
    return mat4 {
        2 / (w), 0, 0, 0,
        0, 2 / h, 0, 0,
        0, 0, 1 / zd, 0,
        x, y, zn / zd, 1,
    };
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

mat4 mat4::operator*(const mat4& _) const {
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
vec4 slerp(const vec4& x, const vec4& y, const float t) {
    auto d = dot(x, y);
    auto w = acosf(std::min(abs(d), 1.0f));
    if (w < 0.001) { return y; }
    return (sinf((1 - t) * w) * (d < 0 ? -1 : 1) * x + sinf(t * w) * y) / sinf(w);
}

vec2 min(const vec2& u, const vec2& v) {
    return vec2 {
        std::min(u.x, v.x),
        std::min(u.y, v.y)
    };
}
vec3 min(const vec3& u, const vec3& v) {
    return vec3 {
        std::min(u.x, v.x),
        std::min(u.y, v.y),
        std::min(u.z, v.z)
    };
}
vec4 min(const vec4& u, const vec4& v) {
    return vec4 {
        std::min(u.x, v.x),
        std::min(u.y, v.y),
        std::min(u.z, v.z),
        std::min(u.w, v.w)
    };
}
vec2 max(const vec2& u, const vec2& v) {
    return vec2 {
        std::max(u.x, v.x),
        std::max(u.y, v.y)
    };
}
vec3 max(const vec3& u, const vec3& v) {
    return vec3 {
        std::max(u.x, v.x),
        std::max(u.y, v.y),
        std::max(u.z, v.z)
    };
}
vec4 max(const vec4& u, const vec4& v) {
    return vec4 {
        std::max(u.x, v.x),
        std::max(u.y, v.y),
        std::max(u.z, v.z),
        std::max(u.w, v.w)
    };
}

mat4 transpose(const mat4& m) {
    return {
        m.m[0], m.m[4], m.m[8], m.m[12],
        m.m[1], m.m[5], m.m[9], m.m[13],
        m.m[2], m.m[6], m.m[10],m.m[14],
        m.m[3], m.m[7], m.m[11],m.m[15],
    };
}

mat4 inverse(const mat4& m) {
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
        v.x * m.m[0] + v.y * m.m[4]  + m.m[8]  + m.m[12],
        v.x * m.m[1] + v.y * m.m[5]  + m.m[9]  + m.m[13]
    };
}
vec3 mul(const vec3& v, const mat4& m) {
    return vec3 {
        v.x * m.m[0] + v.y * m.m[4]  + v.z * m.m[8]  + m.m[12],
        v.x * m.m[1] + v.y * m.m[5]  + v.z * m.m[9]  + m.m[13],
        v.x * m.m[2] + v.y * m.m[6]  + v.z * m.m[10] + m.m[14]
    };
}
vec4 mul(const vec4& v, const mat4& m) {
    return vec4 {
        v.x * m.m[0] + v.y * m.m[4]  + v.z * m.m[8]  + v.w * m.m[12],
        v.x * m.m[1] + v.y * m.m[5]  + v.z * m.m[9]  + v.w * m.m[13],
        v.x * m.m[2] + v.y * m.m[6]  + v.z * m.m[10] + v.w * m.m[14],
        v.x * m.m[3] + v.y * m.m[7]  + v.z * m.m[11] + v.w * m.m[15],
    };
}
vec2 mul_norm(const vec2& v, const mat4& m) {
    return vec2 {
        v.x * m.m[0] + v.y * m.m[4],
        v.x * m.m[1] + v.y * m.m[5]
    };
}
vec3 mul_norm(const vec3& v, const mat4& m) {
    return vec3 {
        v.x * m.m[0] + v.y * m.m[4]  + v.z * m.m[8],
        v.x * m.m[1] + v.y * m.m[5]  + v.z * m.m[9],
        v.x * m.m[2] + v.y * m.m[6]  + v.z * m.m[10]
    };
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
