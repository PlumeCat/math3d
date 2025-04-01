#define ENABLE_TEST
#include <jlib/test_framework.h>
#include "math3d.h"

struct asserter {};
#define assert asserter() <<=

TEST("vec2 default ctor") {
    ASSERT(vec2().x == 0)
    ASSERT(vec2().y == 0)
}
TEST("vec2 brace init") {
    ASSERT(vec2 { 1, 2 }.x == 1)
    ASSERT(vec2 { 3, 4 }.y == 4)
}
TEST("vec2 scalar ctor[1]") {
    ASSERT(vec2(1).x == 1)
    ASSERT(vec2(2).y == 2)
}
TEST("vec2 scalar ctor[2]") {
    ASSERT(vec2(3, 4).x == 3)
    ASSERT(vec2(5, 6).y == 6)
}
TEST("vec2 copy ctor") {
    vec2 u { 10, 20 };
    vec2 v { u };
    ASSERT(v.x == 10)
    ASSERT(v.y == 20)
}
TEST("vec2 move ctor") {
    auto u = vec2 { 50, 60 };
    auto v = vec2(std::move(u));
    ASSERT(v.x == 50);
    ASSERT(v.y == 60);
}
TEST("vec2 alt copy ctor") {
    auto u = ivec2 { 1, 2 };
    auto v = vec2(u);
    ASSERT(v.x == 1);
    ASSERT(v.y == 2);
}
TEST("vec2 copy assign") {
    auto u = vec2 { 100, 200 };
    auto v = vec2 {};
    v = u;
    ASSERT(v.x == 100);
    ASSERT(v.y == 200);
}
TEST("vec2 move assign") {
    auto u = vec2 { 400, 500 };
    auto v = vec2 {};
    v = std::move(u);
    ASSERT(v.x == 400);
    ASSERT(v.y == 500);
}

TEST("vec2 equality") {
    auto u = vec2 { 9, 0 };
    auto v = vec2 { 9, 0 };
    ASSERT(u == v);
    ASSERT(!(u != v));
}
TEST("vec2 format") {
    std::stringstream v;
    v << vec2 { 20,30 };
    ASSERT(v.str() == "vec2 { 20, 30 }");
}

TEST("vec2 assign-add") {
    auto v = vec2 { 20, 20 };
    v += vec2 { 10, 20 };
    ASSERT(v == vec2 { 30, 40 });
}
TEST("vec2 assign-sub") {
    auto v = vec2 { 10, 5 };
    v -= vec2 { 5, 10 };
    ASSERT(v == vec2 { 5, -5 });
}
TEST("vec2 assign-mul") {
    auto v = vec2 { 4, 8 };
    v *= vec2 { 2, 4 };
    ASSERT(v == vec2 { 8, 32 });
}
TEST("vec2 assign-div") {
    auto v = vec2 { 10, 20 };
    v /= 5;
    ASSERT(v == vec2 { 2, 4 });
}
TEST("vec2 assign-add scalar") {
    auto v = vec2 { 5, 6 };
    v += 6;
    ASSERT(v == vec2 { 11, 12 });
}
TEST("vec2 assign-sub scalar") {
    auto v = vec2 { 4, 5 };
    v -= 10;
    ASSERT(v == vec2 { -6, -5 });
}
TEST("vec2 assign-mul scalar") {}
TEST("vec2 assign-div scalar") {}

TEST("vec2 unary sub") {}
TEST("vec2 unary add") {}
TEST("vec2 binary add") {}
TEST("vec2 binary sub") {}
TEST("vec2 binary mul") {}
TEST("vec2 binary div") {}

TEST("vec2 scalar add") {}
TEST("vec2 scalar sub") {}
TEST("vec2 scalar mul") {}
TEST("vec2 scalar div") {}
TEST("vec2 scalar add-R") {}
TEST("vec2 scalar sub-R") {}
TEST("vec2 scalar mul-R") {}
TEST("vec2 scalar div-R") {}

/*
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
*/