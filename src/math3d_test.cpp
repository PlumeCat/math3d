#define JMATH_IMPLEMENTATION
// #define JMATH_ENABLE_SSE2
#include "math3d.h"
// #include <jlib/log.h>

#define ENABLE_TEST
#include <jlib/test_framework.h>
#include <iostream>


TEST("determinant") {
    // ASSERT(determinant(mat4 {
    //     {2,0,0,0},
    //     {0,1,0,0},
    //     {0,0,1,0},
    //     {0,0,0,1}
    // }) == 2);
}
// TEST("determinant 2") {
//     ASSERT(determinant(mat4{{
//         { 5, -7, 2, 2 },
//         { 0, 3, 0, -4 },
//         { -5, -8, 0, 3 },
//         { 0, 5, 0, -6 }
//     }}) == 20);
// }

TEST("vec2 mul mat4") {
    ASSERT(mul(vec2(5, 10), mat4::scale(2)) == vec2(10, 20));
    ASSERT(mul(vec2(5, 10), mat4::translate({ 5, 6, 7 })) == vec2(10, 16));
    ASSERT(mul(vec2(5, 10), mat4::scale(2) * mat4::translate({ 5, 6, 7 })) == vec2(15, 26));
}

TEST("vec3-mat4 mul") {
    ASSERT(mul(vec3(1, 2, 3), mat4::scale(2)) == vec3(2, 4, 6));
    ASSERT(mul(vec3(1, 2, 3), mat4::translate({ 1, 2, 3 })) == vec3(2, 4, 6));
}

TEST("vec3 mul_norm") {
    ASSERT(mul_norm(vec3(1, 2, 3), mat4::scale(2)) == vec3(2, 4, 6));
    auto res = mul_norm(vec3(1, 2, 3), mat4::rotate_x(pi/2.f));
    auto test = vec3 { 1, 3, -2 };
    // ASSERT(res =~ test);
    ASSERT(mul_norm(vec3(1, 2, 3), mat4::translate({ 10, 10, 10 })) == vec3(1, 2, 3));
}


TEST("disabled negation for uvec") {
    // auto test1 = uvec3(1, 2, 3);
    // auto test2 = uvec3(5, 6, 7);
    // auto test3 = test2 - test1;
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

