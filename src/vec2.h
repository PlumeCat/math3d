#pragma once

namespace jm {
    template<arithmetic Type>
    struct vec2 {
        Type x, y;

        vec2(): x(0), y(0) {}
        vec2(Type t): x(t), y(t) {}
        vec2(Type x, Type y): x(x), y(y) {}
        template<arithmetic Other> vec2(const vec2<Other>& other): x(other.x), y(other.y) {}
        vec2(const vec2&) = default;
        vec2(vec2&&) = default;
        
        vec2& operator=(const vec2&) = default;
        vec2& operator=(vec2&&) = default;

        vec2& operator += (const vec2& v) { *this = *this + v; return *this; }
        vec2& operator -= (const vec2& v) { *this = *this - v; return *this; }
        vec2& operator *= (const vec2& v) { *this = *this * v; return *this; }
        vec2& operator /= (const vec2& v) { *this = *this / v; return *this; }
        vec2& operator *= (const Type f) { *this = *this * f; return *this; }
        vec2& operator /= (const Type f) { *this = *this / f; return *this; }

        // "=~" operator for float
        using float_enabled = std::enable_if_t<std::is_floating_point_v<Type>, Type>;
        template<typename = float_enabled> struct epsilon_comparator { Type x, y, epsilon; };
        template<typename = float_enabled> bool operator=(const epsilon_comparator<Type>& r) {
            return
                (abs(x - r.x) < epsilon<Type>()) &&
                (abs(y - r.y) < epsilon<Type>());
        }
        template<typename = float_enabled> epsilon_comparator<Type> operator~() { return { x, y }; }
    };

    template<arithmetic T> vec2<T> operator - (const vec2<T>& v) { return { -v.x, -v.y }; }
    template<arithmetic T> vec2<T> operator + (const vec2<T>& v) { return { +v.x, +v.y }; }
    
    template<arithmetic T, arithmetic S> vec2<T> operator + (const vec2<T>& l, S s) { return vec2<T> { l.x + (T)s, l.y + (T)s }; }
    template<arithmetic T, arithmetic S> vec2<T> operator - (const vec2<T>& l, S s) { return vec2<T> { l.x - (T)s, l.y - (T)s }; }
    template<arithmetic T, arithmetic S> vec2<T> operator * (const vec2<T>& l, S s) { return vec2<T> { l.x * (T)s, l.y * (T)s }; }
    template<arithmetic T, arithmetic S> vec2<T> operator / (const vec2<T>& l, S s) { return vec2<T> { l.x / (T)s, l.y / (T)s }; }

    template<arithmetic T, arithmetic S> vec2<T> operator + (S s, const vec2<T>& l) { return vec2<T> { (T)s + l.x, (T)s + l.y }; }
    template<arithmetic T, arithmetic S> vec2<T> operator - (S s, const vec2<T>& l) { return vec2<T> { (T)s - l.x, (T)s - l.y }; }
    template<arithmetic T, arithmetic S> vec2<T> operator * (S s, const vec2<T>& l) { return vec2<T> { (T)s * l.x, (T)s * l.y }; }
    template<arithmetic T, arithmetic S> vec2<T> operator / (S s, const vec2<T>& l) { return vec2<T> { (T)s / l.x, (T)s / l.y }; }

    template<arithmetic T> vec2<T> operator + (const vec2<T>& l, const vec2<T>& r) { return vec2<T> { l.x + r.x, l.y + r.y }; }
    template<arithmetic T> vec2<T> operator - (const vec2<T>& l, const vec2<T>& r) { return vec2<T> { l.x - r.x, l.y - r.y }; }
    template<arithmetic T> vec2<T> operator * (const vec2<T>& l, const vec2<T>& r) { return vec2<T> { l.x * r.x, l.y * r.y }; }
    template<arithmetic T> vec2<T> operator / (const vec2<T>& l, const vec2<T>& r) { return vec2<T> { l.x / r.x, l.y / r.y }; }

    template<arithmetic T> bool operator == (const vec2<T>& l, const vec2<T>& r) { return (l.x == r.x) && (l.y == r.y); }
    template<arithmetic T> std::ostream& operator << (std::ostream& o, const vec2<T>& v) {
        return o << "vec2 { " << v.x << ", " << v.y << " }";
    }
};

