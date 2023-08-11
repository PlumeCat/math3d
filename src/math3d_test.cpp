#define JMATH_IMPLEMENTATION
// #define JMATH_ENABLE_SSE2
#include "math3d.h"
#include <jlib/log.h>

#define ENABLE_TEST
#include <jlib/test_framework.h>
#include <iostream>

TEST("vec2 add") {
    ASSERT(vec2(1, 2) + vec2(3, 4) == vec2(4, 6));
}
TEST("vec2 sub") {
    ASSERT(vec2(3, 4) - vec2(1, 2) == vec2(2, 2));
}


// TEST("determinant") {
//     ASSERT(determinant(mat4{{
//         {2,0,0,0},
//         {0,1,0,0},
//         {0,0,1,0},
//         {0,0,0,1}
//     }}) == 2);
// }
// TEST("determinant 2") {
//     ASSERT(determinant(mat4{{
//         { 5, -7, 2, 2 },
//         { 0, 3, 0, -4 },
//         { -5, -8, 0, 3 },
//         { 0, 5, 0, -6 }
//     }}) == 20);
// }

TEST("inverse scale") {
    auto scale = mat4::scale({ 1, 2, 5 });
    auto inv = inverse(scale);
    log(inv);
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

    std::cout << "testvec: " << testvec << std::endl;
    log("view:    ", view);
    log("inv.view:", inverse_view);
    log("ident:   ", ident_hopefully);

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

    log(proj);
    log(inverse_proj);
    log(ident);

    ASSERT(abs(testvec.x) < 0.0001);
    ASSERT(abs(testvec.y) < 0.0001);
    ASSERT(abs(testvec.z) < 0.0001);
}

IMPLEMENT_TESTS()
int main(int argc, char* argv[]) {
    log("RUNNING TESTS");
    RUN_TESTS()
}

