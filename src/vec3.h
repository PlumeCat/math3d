#pragma once

namespace jm {
    template<arithmetic Type>
    struct vec3 {
        struct epsilon_comparator { Type x, y, z, epsilon; };
        Type x, y, z;

        vec3() { x = 0; y = 0; z = 0; }
        vec3(Type t): x(t), y(t), z(t) {}
        vec3(Type x, Type y, Type z): x(x), y(y), z(z) {}
        template<arithmetic Other> vec3(const vec3<Other>& other): x(other.x), y(other.y), z(other.z) {}
        vec3(const vec3&) = default;
        vec3(vec3&&) = default;
        
        bool operator=(const epsilon_comparator& e) {
            return
                (abs(e.x - x) < e.epsilon) &&
                (abs(e.y - y) < e.epsilon) &&
                (abs(e.z - z) < e.epsilon);
        }
        vec3& operator=(const vec3&) = default;
        vec3& operator=(vec3&&) = default;

        vec3& operator += (const vec3& o) { *this = *this + o; return *this; }
        vec3& operator -= (const vec3& o) { *this = *this - o; return *this; }
        vec3& operator *= (const vec3& v) { *this = *this * v; return *this; }
        vec3& operator /= (const vec3& v) { *this = *this / v; return *this; }
        vec3& operator *= (const Type f) { *this = *this * f; return *this; }
        vec3& operator /= (const Type f) { *this = *this / f; return *this; }
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

    template<arithmetic T> bool operator == (const vec3<T>& l, const vec3<T>& r) {
        return
            (l.x == r.x) &&
            (l.y == r.y) &&
            (l.z == r.z);
    }
    template<arithmetic T> std::ostream& operator << (std::ostream& o, const vec3<T>& v) {
        return o << "vec3 { " << v.x << ", " << v.y << ", " << v.z << " }";
    }

};

using vec3 = jm::vec3<float>;
using ivec3 = jm::vec3<int32_t>;
using uvec3 = jm::vec3<uint32_t>;

#ifdef JMATH_IMPLEMENTATION
jm::vec3<float>::epsilon_comparator operator~(const jm::vec3<float>& l) { return { l.x, l.y, l.z, epsilon_f }; }
jm::vec3<double>::epsilon_comparator operator~(const jm::vec3<double>& l) { return { l.x, l.y, l.z, epsilon_d }; }
#endif