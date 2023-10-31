#pragma once

namespace jm {
    template<arithmetic Type>
    struct vec4 {
        Type x, y, z, w;

        vec4() { x = 0; y = 0; z = 0; w = 0; }
        vec4(Type t): x(t), y(t), z(t), w(t) {}
        vec4(Type x, Type y, Type z, Type w): x(x), y(y), z(z), w(w) {}
        vec4(const vec4&) = default;
        vec4(vec4&&) = default;
        vec4& operator=(const vec4&) = default;
        vec4& operator=(vec4&&) = default;

        vec4& operator += (const vec4& o) { *this = *this + o; return *this; }
        vec4& operator -= (const vec4& o) { *this = *this - o; return *this; }
        vec4& operator *= (const vec4& v) { *this = *this * v; return *this; }
        vec4& operator /= (const vec4& v) { *this = *this / v; return *this; }
        vec4& operator *= (const Type f) { *this = *this * f; return *this; }
        vec4& operator /= (const Type f) { *this = *this / f; return *this; }
    };

    template<arithmetic T> vec4<T> operator - (const vec4<T>& v) { return { -v.x, -v.y, -v.z, -v.w }; }
    template<arithmetic T> vec4<T> operator + (const vec4<T>& v) { return { +v.x, +v.y, +v.z, +v.w }; }
    
    template<arithmetic T, arithmetic S> vec4<T> operator + (const vec4<T>& l, S s) { return { l.x + (T)s, l.y + (T)s, l.z + (T)s, l.w + (T)s }; }
    template<arithmetic T, arithmetic S> vec4<T> operator - (const vec4<T>& l, S s) { return { l.x - (T)s, l.y - (T)s, l.z - (T)s, l.w - (T)s }; }
    template<arithmetic T, arithmetic S> vec4<T> operator * (const vec4<T>& l, S s) { return { l.x * (T)s, l.y * (T)s, l.z * (T)s, l.w * (T)s }; }
    template<arithmetic T, arithmetic S> vec4<T> operator / (const vec4<T>& l, S s) { return { l.x / (T)s, l.y / (T)s, l.z / (T)s, l.w / (T)s }; }

    template<arithmetic T, arithmetic S> vec4<T> operator + (S s, const vec4<T>& r) { return { (T)s + r.x, (T)s + r.y, (T)s + r.z, (T)s + r.w }; }
    template<arithmetic T, arithmetic S> vec4<T> operator - (S s, const vec4<T>& r) { return { (T)s - r.x, (T)s - r.y, (T)s - r.z, (T)s - r.w }; }
    template<arithmetic T, arithmetic S> vec4<T> operator * (S s, const vec4<T>& r) { return { (T)s * r.x, (T)s * r.y, (T)s * r.z, (T)s * r.w }; }
    template<arithmetic T, arithmetic S> vec4<T> operator / (S s, const vec4<T>& r) { return { (T)s / r.x, (T)s / r.y, (T)s / r.z, (T)s / r.w }; }

    template<arithmetic T> vec4<T> operator + (const vec4<T>& l, const vec4<T>& r) { return { l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w }; }
    template<arithmetic T> vec4<T> operator - (const vec4<T>& l, const vec4<T>& r) { return { l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w }; }
    template<arithmetic T> vec4<T> operator * (const vec4<T>& l, const vec4<T>& r) { return { l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w }; }
    template<arithmetic T> vec4<T> operator / (const vec4<T>& l, const vec4<T>& r) { return { l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w }; }

    template<arithmetic T> bool operator == (const vec4<T>& l, const vec4<T>& r) { return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w; }

    template<arithmetic T> std::ostream& operator << (std::ostream& o, const vec4<T>& v) {
        return o << "vec4 { " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}";
    }
};

using vec4 = jm::vec4<float>;
using ivec4 = jm::vec4<int32_t>;
using uvec4 = jm::vec4<uint32_t>;