#pragma once

namespace jm {
    template<scalar Type>
    struct vec3 {

        Type x, y, z;

        vec3(): x(0), y(0), z(0) {}
        vec3(scalar auto t): x(t), y(t), z(t) {}
        vec3(scalar auto x, scalar auto y, scalar auto z): x(x), y(y), z(z) {}
        vec3(const vec3& other): x(other.x), y(other.y), z(other.z) {}
        template<typename U> explicit vec3(const vec3<U>& other): x(other.x), y(other.y), z(other.z) {}
        vec3(vec3&&) = default;
        vec3& operator=(const vec3& other) { x = other.x; y = other.y; z = other.z; return *this; }
        vec3& operator=(vec3&&) = default;

        vec3& operator += (const vec3& o) { *this = *this + o; return *this; }
        vec3& operator -= (const vec3& o) { *this = *this - o; return *this; }
        vec3& operator *= (const vec3& v) { *this = *this * v; return *this; }
        vec3& operator /= (const vec3& v) { *this = *this / v; return *this; }

        vec3& operator += (scalar auto f) { *this = *this + f; return *this; }
        vec3& operator -= (scalar auto f) { *this = *this - f; return *this; }
        vec3& operator *= (scalar auto f) { *this = *this * f; return *this; }
        vec3& operator /= (scalar auto f) { *this = *this / f; return *this; }

    };

    template<typename X, typename Y, typename Z> vec3(X, Y, Z) -> vec3<X>; // CTAD guide for 3-argument constructor
    template<typename T> concept IsVec3 = IsSpecializationOf<T, vec3>;

    auto operator - (const IsVec3 auto& v) { return vec3 { -v.x, -v.y, -v.z }; }
    auto operator + (const IsVec3 auto& v) { return vec3 { +v.x, +v.y, +v.z }; }

    auto operator + (const IsVec3 auto& l, scalar auto s) { return vec3 { l.x + s, l.y + s, l.z + s }; }
    auto operator - (const IsVec3 auto& l, scalar auto s) { return vec3 { l.x - s, l.y - s, l.z - s }; }
    auto operator * (const IsVec3 auto& l, scalar auto s) { return vec3 { l.x * s, l.y * s, l.z * s }; }
    auto operator / (const IsVec3 auto& l, scalar auto s) { return vec3 { l.x / s, l.y / s, l.z / s }; }
    auto operator + (scalar auto s, const IsVec3 auto& l) { return vec3 { s + l.x, s + l.y, s + l.z }; }
    auto operator - (scalar auto s, const IsVec3 auto& l) { return vec3 { s - l.x, s - l.y, s - l.z }; }
    auto operator * (scalar auto s, const IsVec3 auto& l) { return vec3 { s * l.x, s * l.y, s * l.z }; }
    auto operator / (scalar auto s, const IsVec3 auto& l) { return vec3 { s / l.x, s / l.y, s / l.z }; }

    auto operator + (const IsVec3 auto& l, const IsVec3 auto& r) { return vec3 { l.x + r.x, l.y + r.y, l.z + r.z }; }
    auto operator - (const IsVec3 auto& l, const IsVec3 auto& r) { return vec3 { l.x - r.x, l.y - r.y, l.z - r.z }; }
    auto operator * (const IsVec3 auto& l, const IsVec3 auto& r) { return vec3 { l.x * r.x, l.y * r.y, l.z * r.z }; }
    auto operator / (const IsVec3 auto& l, const IsVec3 auto& r) { return vec3 { l.x / r.x, l.y / r.y, l.z / r.z }; }

    auto operator == (const IsVec3 auto& l, const IsVec3 auto& r) { return l.x == r.x && l.y == r.y && l.z == r.z; }
    std::ostream& operator << (std::ostream& o, const IsVec3 auto& v) {
        return o << "vec3 { " << v.x << ", " << v.y << ", " << v.z << " }";
    }
};

using vec3 = jm::vec3<float>;
using ivec3 = jm::vec3<int32_t>;
using uvec3 = jm::vec3<uint32_t>;

#ifdef JMATH_IMPLEMENTATION
// struct epsilon_comparator { Type x, y, z, epsilon; };
// bool operator=(const epsilon_comparator& e) {
//     return
//         (abs(e.x - x) < e.epsilon) &&
//         (abs(e.y - y) < e.epsilon) &&
//         (abs(e.z - z) < e.epsilon);
// }
// jm::vec3<float>::epsilon_comparator operator~(const jm::vec3<float>& l) { return { l.x, l.y, l.z, epsilon_f }; }
// jm::vec3<double>::epsilon_comparator operator~(const jm::vec3<double>& l) { return { l.x, l.y, l.z, epsilon_d }; }
#endif