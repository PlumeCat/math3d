# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.23)

project ("math3d")
message("CMake Project Folder: ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source to this project's executable.
include_directories(math3d_test "${PROJECT_SOURCE_DIR}/../jlib/")
# add_library(math3d "src/math3d.cpp" "src/math3d.h")
add_executable(math3d_test "src/math3d_test.cpp")
# target_link_libraries(math3d_test math3d)

message("CMake Build Type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL Release)
	message("RELEASE BUILD")
	add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
elseif (CMAKE_BUILD_TYPE STREQUAL Debug)
	message("DEBUG BUILD")
	add_definitions(-D_ITERATOR_DEBUG_LEVEL=2)
else()
	message("WARNING: UNKNOWN BUILD TYPE <${CMAKE_BUILD_TYPE}>")
endif()

