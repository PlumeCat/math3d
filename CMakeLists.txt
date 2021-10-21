# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("math3d")

# Add source to this project's executable.
add_library (math3d "src/math3d.cpp" "src/math3d.h")

