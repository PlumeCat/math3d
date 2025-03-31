#pragma once

namespace jm {
    template<scalar Type>
    struct vec2 {
        Type x, y;

        vec2(): x(0), y(0) {}
        vec2(scalar auto t): x(t), y(t) {}
        vec2(scalar auto x, scalar auto y): x(x), y(y) {}
        vec2(const vec2& other): x(other.x), y(other.y) {}
        template<typename U> explicit vec2(const vec2<U>& other): x(other.x), y(other.y) {}
        vec2(vec2&&) = default;
        vec2& operator=(const vec2& other) { x = other.x; y = other.y; return *this; }
        vec2& operator=(vec2&&) = default;

        vec2& operator += (const vec2& v) { *this = *this + v; return *this; }
        vec2& operator -= (const vec2& v) { *this = *this - v; return *this; }
        vec2& operator *= (const vec2& v) { *this = *this * v; return *this; }
        vec2& operator /= (const vec2& v) { *this = *this / v; return *this; }

        vec2& operator += (scalar auto f) { *this = *this + f; return *this; }
        vec2& operator -= (scalar auto f) { *this = *this - f; return *this; }
        vec2& operator *= (scalar auto f) { *this = *this * f; return *this; }
        vec2& operator /= (scalar auto f) { *this = *this / f; return *this; }

    };

    template<typename X, typename Y> vec2(X, Y) -> vec2<X>; // CTAD guide for 2-argument constructor
    template<typename T> concept IsVec2 = IsSpecializationOf<T, jm::vec2>;

    auto operator - (const IsVec2 auto& v) { return jm::vec2 { -v.x, -v.y }; }
    auto operator + (const IsVec2 auto& v) { return jm::vec2 { +v.x, +v.y }; }

    auto operator + (const IsVec2 auto& l, scalar auto s) { return jm::vec2 { l.x + s, l.y + s }; }
    auto operator - (const IsVec2 auto& l, scalar auto s) { return jm::vec2 { l.x - s, l.y - s }; }
    auto operator * (const IsVec2 auto& l, scalar auto s) { return jm::vec2 { l.x * s, l.y * s }; }
    auto operator / (const IsVec2 auto& l, scalar auto s) { return jm::vec2 { l.x / s, l.y / s }; }
    auto operator + (scalar auto s, const IsVec2 auto& l) { return jm::vec2 { s + l.x, s + l.y }; }
    auto operator - (scalar auto s, const IsVec2 auto& l) { return jm::vec2 { s - l.x, s - l.y }; }
    auto operator * (scalar auto s, const IsVec2 auto& l) { return jm::vec2 { s * l.x, s * l.y }; }
    auto operator / (scalar auto s, const IsVec2 auto& l) { return jm::vec2 { s / l.x, s / l.y }; }

    auto operator + (const IsVec2 auto& l, const IsVec2 auto& r) { return jm::vec2 { l.x + r.x, l.y + r.y }; }
    auto operator - (const IsVec2 auto& l, const IsVec2 auto& r) { return jm::vec2 { l.x - r.x, l.y - r.y }; }
    auto operator * (const IsVec2 auto& l, const IsVec2 auto& r) { return jm::vec2 { l.x * r.x, l.y * r.y }; }
    auto operator / (const IsVec2 auto& l, const IsVec2 auto& r) { return jm::vec2 { l.x / r.x, l.y / r.y }; }

    auto operator == (const IsVec2 auto& l, const IsVec2 auto& r) { return l.x == r.x && l.y == r.y; }

    std::ostream& operator << (std::ostream& o, const IsVec2 auto& v) {
        return o << "vec2 { " << v.x << ", " << v.y << " }";
    }
};

