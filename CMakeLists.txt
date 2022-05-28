# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("math3d")

# Add source to this project's executable.
add_library (math3d "src/math3d.cpp" "src/math3d.h")

message("Cmake Build Type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL Release)
	message("RELEASE BUILD")
	add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
elseif (CMAKE_BUILD_TYPE STREQUAL Debug)
	message("DEBUG BUILD")
	add_definitions(-D_ITERATOR_DEBUG_LEVEL=2)
else()
	message("WARNING: UNKNOWN BUILD TYPE <${CMAKE_BUILD_TYPE}>")
endif()

