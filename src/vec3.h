#pragma once

namespace jm {
    template<arithmetic Type>
    struct vec3 {
        Type x, y, z;

        vec3(): x(0), y(0), z(0) {}
        vec3(Type t): x(t), y(t), z(t) {}
        vec3(Type x, Type y, Type z): x(x), y(y), z(z) {}
        template<arithmetic Other> vec3(const vec3<Other>& other): x(other.x), y(other.y), z(other.z) {}
        vec3(const vec3&) = default;
        vec3(vec3&&) = default;

        vec3& operator=(const vec3&) = default;
        vec3& operator=(vec3&&) = default;

        vec3& operator += (const vec3& o) { *this = *this + o; return *this; }
        vec3& operator -= (const vec3& o) { *this = *this - o; return *this; }
        vec3& operator *= (const vec3& v) { *this = *this * v; return *this; }
        vec3& operator /= (const vec3& v) { *this = *this / v; return *this; }
        vec3& operator *= (const Type f) { *this = *this * f; return *this; }
        vec3& operator /= (const Type f) { *this = *this / f; return *this; }

        // "=~" operator for float
        using float_enabled = std::enable_if_t<std::is_floating_point_v<Type>, Type>;
        template<typename = float_enabled> struct epsilon_comparator { Type x, y, z; };
        template<typename = float_enabled> bool operator=(const epsilon_comparator<Type>& r) {
            return
                (abs(x - r.x) < epsilon<Type>()) &&
                (abs(y - r.y) < epsilon<Type>()) &&
                (abs(z - r.z) < epsilon<Type>());
        }
        template<typename = float_enabled> epsilon_comparator<Type> operator~() { return { x, y, z }; }
    };

    template<arithmetic T> vec3<T> operator - (const vec3<T>& v) { return { -v.x, -v.y, -v.z }; }
    template<arithmetic T> vec3<T> operator + (const vec3<T>& v) { return { +v.x, +v.y, +v.z }; }

    template<arithmetic T, arithmetic S> vec3<T> operator + (const vec3<T>& l, S s) { return vec3<T> { l.x + (T)s, l.y + (T)s, l.z + (T)s }; }
    template<arithmetic T, arithmetic S> vec3<T> operator - (const vec3<T>& l, S s) { return vec3<T> { l.x - (T)s, l.y - (T)s, l.z - (T)s }; }
    template<arithmetic T, arithmetic S> vec3<T> operator * (const vec3<T>& l, S s) { return vec3<T> { l.x * (T)s, l.y * (T)s, l.z * (T)s }; }
    template<arithmetic T, arithmetic S> vec3<T> operator / (const vec3<T>& l, S s) { return vec3<T> { l.x / (T)s, l.y / (T)s, l.z / (T)s }; }

    template<arithmetic T, arithmetic S> vec3<T> operator + (S s, const vec3<T>& l) { return vec3<T> { (T)s + l.x, (T)s + l.y, (T)s + l.z }; }
    template<arithmetic T, arithmetic S> vec3<T> operator - (S s, const vec3<T>& l) { return vec3<T> { (T)s - l.x, (T)s - l.y, (T)s - l.z }; }
    template<arithmetic T, arithmetic S> vec3<T> operator * (S s, const vec3<T>& l) { return vec3<T> { (T)s * l.x, (T)s * l.y, (T)s * l.z }; }
    template<arithmetic T, arithmetic S> vec3<T> operator / (S s, const vec3<T>& l) { return vec3<T> { (T)s / l.x, (T)s / l.y, (T)s / l.z }; }

    template<arithmetic T> vec3<T> operator + (const vec3<T>& l, const vec3<T>& r) { return vec3<T> { l.x + r.x, l.y + r.y, l.z + r.z }; }
    template<arithmetic T> vec3<T> operator - (const vec3<T>& l, const vec3<T>& r) { return vec3<T> { l.x - r.x, l.y - r.y, l.z - r.z }; }
    template<arithmetic T> vec3<T> operator * (const vec3<T>& l, const vec3<T>& r) { return vec3<T> { l.x * r.x, l.y * r.y, l.z * r.z }; }
    template<arithmetic T> vec3<T> operator / (const vec3<T>& l, const vec3<T>& r) { return vec3<T> { l.x / r.x, l.y / r.y, l.z / r.z }; }

    template<arithmetic T> bool operator == (const vec3<T>& l, const vec3<T>& r) { return (l.x == r.x) && (l.y == r.y) && (l.z == r.z); }
    template<arithmetic T> std::ostream& operator << (std::ostream& o, const vec3<T>& v) {
        return o << "vec3 { " << v.x << ", " << v.y << ", " << v.z << " }";
    }
};
