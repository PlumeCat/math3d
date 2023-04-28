#define JMATH_IMPLEMENTATION
// #define JMATH_ENABLE_SSE2
#include "math3d.h"
#include <jlib/log.h>
#define ENABLE_TEST
#include <jlib/test_framework.h>

#include <iostream>
// using namespace std;

TEST("vec2 add") {
    ASSERT(vec2(1, 2) + vec2(3, 4) == vec2(4, 6));
}
TEST("vec2 sub") {
    ASSERT(vec2(3, 4) - vec2(1, 2) == vec2(2, 2));
}
TEST("determinant") {
    ASSERT(determinant(mat4{{
        {2,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
        {0,0,0,1}
    }}) == 2);
}
TEST("determinant 2") {
    ASSERT(determinant(mat4{{
        { 5, -7, 2, 2 },
        { 0, 3, 0, -4 },
        { -5, -8, 0, 3 },
        { 0, 5, 0, -6 }
    }}) == 20);
}

uint64_t first_52_bits(uint64_t n) {
    return n & (1 << 53 - 1);
}

double make_double(bool sign_bit, uint16_t exponent, uint64_t frac) {
    auto data = uint64_t { 0 };
    data |= uint64_t(sign_bit) << 63;
    data |= first_52_bits(frac);

    log<true>(first_52_bits(INT64_MAX));

    return 0;
}


TEST("double trouble") {
    make_double(true, 1, 1);
}


IMPLEMENT_TESTS()
int main(int argc, char* argv[]) {
    log("RUNNING TESTS");
    RUN_TESTS()
}

