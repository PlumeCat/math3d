#pragma once

#include <cmath>
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

float distance(const vec2& a, const vec2& b);
float distance(const vec3& a, const vec3& b);
float distance(const vec4& a, const vec4& b);

vec2 lerp(const vec2& x, const vec2& y, const float t);
vec3 lerp(const vec3& x, const vec3& y, const float t);
vec4 lerp(const vec4& x, const vec4& y, const float t);

vec2 mul(const vec2& v, const mat4& m);
vec3 mul(const vec3& v, const mat4& m);
vec4 mul(const vec4& v, const mat4& m);
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
#else
#define per_component_2(f) vec2 f(const vec2& v);
#define per_component_3(f) vec3 f(const vec3& v);
#define per_component_4(f) vec4 f(const vec4& v);
#endif
#define per_component_all(func)\
    per_component_2(func)\
    per_component_3(func)\
    per_component_4(func)

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
    //	Result[0][0] = s.x;
    //	Result[1][0] = s.y;
    //	Result[2][0] = s.z;
    //	Result[0][1] = u.x;
    //	Result[1][1] = u.y;
    //	Result[2][1] = u.z;
    //	Result[0][2] = -f.x;
    //	Result[1][2] = -f.y;
    //	Result[2][2] = -f.z;
    //	Result[3][0] = -dot(s, eye);
    //	Result[3][1] = -dot(u, eye);
    //	Result[3][2] = dot(f, eye);
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
        0, 0, zs, zb,
        0, 0, -1, 0
    };
    /*
    * Adapted from GLM source
    
    mat4 perspectiveFovRH_ZO(float fov, float width, float height, float zNear, float zFar) {
        T const rad = fov;
        T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
        T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

        mat<4, 4, T, defaultp> Result(static_cast<T>(0));
        Result[0][0] = w;
        Result[1][1] = h;
        Result[2][2] = zFar / (zNear - zFar);
        Result[2][3] = - static_cast<T>(1);
        Result[3][2] = -(zFar * zNear) / (zFar - zNear);
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
        0, 0, zn / zd, 1
    };
    /*
    Adapted from GLM source
        template<typename T>
        GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_ZO(T left, T right, T bottom, T top, T zNear, T zFar)
        {
            mat<4, 4, T, defaultp> Result(1);
            Result[0][0] = static_cast<T>(2) / (right - left);
            Result[1][1] = static_cast<T>(2) / (top - bottom);
            Result[2][2] = - static_cast<T>(1) / (zFar - zNear);
            Result[3][0] = - (right + left) / (right - left);
            Result[3][1] = - (top + bottom) / (top - bottom);
            Result[3][2] = - zNear / (zFar - zNear);
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
//         m.m[0][0] * determinant3x3<1, 2, 3, 1, 2, 3>(m) +
//         m.m[0][1] * -determinant3x3<1, 2, 3, 0, 2, 3>(m) +
//         m.m[0][2] * determinant3x3<1, 2, 3, 0, 1, 3>(m) +
//         m.m[0][3] * -determinant3x3<1, 2, 3, 0, 1, 2>(m);
// }

#endif