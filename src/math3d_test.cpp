#define JMATH_IMPLEMENTATION
// #define JMATH_ENABLE_SSE2
#include "math3d.h"
// #include <jlib/log.h>

#define ENABLE_TEST
#include <jlib/test_framework.h>
#include <iostream>

TEST("vec2 equality") {
    auto v = vec2(1, 2);
    auto w = vec2(1, 2);
    ASSERT(v == w);
    ASSERT(v == v);
    ASSERT(w == w);
}
TEST("vec2 ctor") {
    auto v = vec2(1, 2);
    ASSERT(v == vec2(1, 2));
    v = vec2(3);
    ASSERT(v == vec2(3));
    v = vec2(vec2(vec2(40)));
    ASSERT(v == vec2(40));
}
TEST("vec2 add") {
    ASSERT(vec2(1, 2) + vec2(3, 4) == vec2(4, 6));
    ASSERT(vec2(1, 2) + 3 == vec2(4, 5));
    ASSERT(5 + vec2(6, 7) == vec2(11, 12));
    auto v = vec2(10, 18);
    v += 4; v += vec2(1, 2);
    ASSERT(v == vec2(15, 24));
}
TEST("vec2 sub") {
    ASSERT(vec2(1, 2) - vec2(3, 4) == vec2(-2, -2));
    ASSERT(vec2(1, 2) - 3 == vec2(-2, -1));
    ASSERT(5 - vec2(6, 7) == vec2(-1, -2));
    auto v = vec2(10, 18);
    v -= 4; v -= vec2(1, 2);
    ASSERT(v == vec2(5, 12));
}
TEST("vec2 mul") {
    ASSERT(vec2(5, 6) * vec2(7, 8) == vec2(35, 48));
    ASSERT(vec2(2.5, 3) * 10 == vec2(25, 30));
    ASSERT(10 * vec2(-65, -66) == vec2(-650, -660));
    auto v = vec2(10, 18);
    v -= 4; v -= vec2(1, 2);
    ASSERT(v == vec2(5, 12));
}
TEST("vec2 div") {
    ASSERT(vec2(8, 4) / vec2(2, 2) == vec2(4, 2));
    ASSERT(vec2(100, 200) / 10.f == vec2(10, 20));
    ASSERT(360 / vec2(180, 12) == vec2(2, 30));
}


TEST("determinant") {
    ASSERT(determinant(mat4 {
        {2,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
        {0,0,0,1}
    }) == 2);
}
// TEST("determinant 2") {
//     ASSERT(determinant(mat4{{
//         { 5, -7, 2, 2 },
//         { 0, 3, 0, -4 },
//         { -5, -8, 0, 3 },
//         { 0, 5, 0, -6 }
//     }}) == 20);
// }

TEST("disabled negation for uvec") {
    auto test1 = uvec3(1, 2, 3);
    auto test2 = uvec3(5, 6, 7);
    auto test3 = test2 - test1;
    log(test3);

    log(-vec2(1, 2));
    log(-ivec2(3, 4));
}

TEST("inverse scale") {
    auto scale = mat4::scale({ 1, 2, 5 });
    auto inv = inverse(scale);
    ASSERT(inv.m[0] == 1.0f);
    ASSERT(inv.m[5] == 0.5f);
    ASSERT(inv.m[10] == 0.2f);

    auto testvec = vec3 { 6, 135, 3465 };
    testvec -= mul(testvec, inv * scale);
    ASSERT(abs(testvec.x) < 0.0001);
    ASSERT(abs(testvec.y) < 0.0001);
    ASSERT(abs(testvec.z) < 0.0001);
}

TEST("inverse view") {
    auto view = mat4::look_at({ 10, 10, 10 }, { 5, 3, 2 }, { 0, 1, 0 });
    auto inverse_view = inverse(view);
    auto ident_hopefully = view * inverse_view;

    auto testvec = vec3 { 1, 3, 2 };
    testvec -= mul(testvec, ident_hopefully);

    ASSERT(abs(testvec.x) < 0.0001);
    ASSERT(abs(testvec.y) < 0.0001);
    ASSERT(abs(testvec.z) < 0.0001);
}

TEST("inverse proj") {
    auto proj = mat4::perspective(pi/3, 1.5, 1, 1024);
    auto inverse_proj = inverse(proj);
    auto ident = proj * inverse_proj;
    auto testvec = vec3 { 183, 34, -125 };
    testvec -= mul(testvec, ident);

    ASSERT(abs(testvec.x) < 0.0001);
    ASSERT(abs(testvec.y) < 0.0001);
    ASSERT(abs(testvec.z) < 0.0001);
}

IMPLEMENT_TESTS()
int main(int argc, char* argv[]) {
    RUN_TESTS()
}

