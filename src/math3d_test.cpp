#include "math3d.h"

#include <iostream>
using namespace std;

#define test(name, input, output)\
    if (input == output) {\
        cout << "Test succeeded: " << name << endl;\
    } else {\
        cout << "Test failed: " << name << endl;\
    }

int main(int argc, char* argv[]) {
    test("vec2 add", vec2(1, 2) + vec2(3, 4), vec2(4, 6));
    test("vec2 sub", vec2(3, 4) - vec2(1, 2), vec2(2, 2));
    // TODO: more tests!
}

