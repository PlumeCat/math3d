#pragma once

namespace jm {
    template<scalar Type>
    struct vec4 {

        Type x, y, z, w;

        vec4(): x(0), y(0), z(0), w(0) {}
        vec4(scalar auto t): x(t), y(t), z(t), w(t) {}
        vec4(scalar auto x, scalar auto y, scalar auto z, scalar auto w): x(x), y(y), z(z), w(w) {}
        vec4(const vec4& other): x(other.x), y(other.y), z(other.z), w(other.w) {}
        template<typename U> explicit vec4(const vec4<U>& other): x(other.x), y(other.y), z(other.z), w(other.w) {}
        vec4(vec4&&) = default;
        vec4& operator=(const vec4& other) { x = other.x; y = other.y; z = other.z; w = other.w; return *this; }
        vec4& operator=(vec4&&) = default;

        vec4& operator += (const vec4& o) { *this = *this + o; return *this; }
        vec4& operator -= (const vec4& o) { *this = *this - o; return *this; }
        vec4& operator *= (const vec4& v) { *this = *this * v; return *this; }
        vec4& operator /= (const vec4& v) { *this = *this / v; return *this; }

        vec4& operator += (scalar auto f) { *this = *this + f; return *this; }
        vec4& operator -= (scalar auto f) { *this = *this - f; return *this; }
        vec4& operator *= (scalar auto f) { *this = *this * f; return *this; }
        vec4& operator /= (scalar auto f) { *this = *this / f; return *this; }

    };

    template<typename X, typename Y, typename Z, typename W> vec4(X, Y, Z, W) -> vec4<X>; // CTAD guide for 4-argument constructor
    template<typename T> concept IsVec4 = IsSpecializationOf<T, jm::vec4>;

    auto operator - (const IsVec4 auto& v) { return vec4 { -v.x, -v.y, -v.z, -v.w }; }
    auto operator + (const IsVec4 auto& v) { return vec4 { +v.x, +v.y, +v.z, +v.w }; }

    auto operator + (const IsVec4 auto& l, scalar auto s) { return vec4 { l.x + s, l.y + s, l.z + s, l.w + s }; }
    auto operator - (const IsVec4 auto& l, scalar auto s) { return vec4 { l.x - s, l.y - s, l.z - s, l.w - s }; }
    auto operator * (const IsVec4 auto& l, scalar auto s) { return vec4 { l.x * s, l.y * s, l.z * s, l.w * s }; }
    auto operator / (const IsVec4 auto& l, scalar auto s) { return vec4 { l.x / s, l.y / s, l.z / s, l.w / s }; }

    auto operator + (scalar auto s, const IsVec4 auto& r) { return vec4 { s + r.x, s + r.y, s + r.z, s + r.w }; }
    auto operator - (scalar auto s, const IsVec4 auto& r) { return vec4 { s - r.x, s - r.y, s - r.z, s - r.w }; }
    auto operator * (scalar auto s, const IsVec4 auto& r) { return vec4 { s * r.x, s * r.y, s * r.z, s * r.w }; }
    auto operator / (scalar auto s, const IsVec4 auto& r) { return vec4 { s / r.x, s / r.y, s / r.z, s / r.w }; }

    auto operator + (const IsVec4 auto& l, const IsVec4 auto& r) { return vec4 { l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w }; }
    auto operator - (const IsVec4 auto& l, const IsVec4 auto& r) { return vec4 { l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w }; }
    auto operator * (const IsVec4 auto& l, const IsVec4 auto& r) { return vec4 { l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w }; }
    auto operator / (const IsVec4 auto& l, const IsVec4 auto& r) { return vec4 { l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w }; }

    auto operator == (const IsVec4 auto& l, const IsVec4 auto& r) { return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w; }

    std::ostream& operator << (std::ostream& o, const IsVec4 auto& v) {
        return o << "vec4 { " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << " }";
    }
};