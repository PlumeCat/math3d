// math3d.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

// TODO: Reference additional headers your program requires here.

#pragma once

const float pi = 3.1415926535f;

struct vec2 {
	float x = 0, y = 0;

	vec2() = default;
	vec2(float x, float y): x(x), y(y) {}
	vec2(const vec2&) = default;
	vec2(vec2&&) = default;
	vec2& operator=(const vec2&) = default;
	vec2& operator=(vec2&&) = default;

	inline vec2 operator - () const { return { -x, -y }; }

	inline vec2& operator += (const vec2& v) { *this = *this + v; return *this; }
	inline vec2& operator -= (const vec2& v) { *this = *this - v; return *this; }
	inline vec2& operator *= (const vec2& v) { *this = *this * v; return *this; }
	inline vec2& operator /= (const vec2& v) { *this = *this / v; return *this; }
	inline vec2& operator *= (const float f) { *this = *this * f; return *this; }
	inline vec2& operator /= (const float f) { *this = *this / f; return *this; }

	inline vec2 operator + (const vec2& r) const { return vec2 { x + r.x, y + r.y }; }
	inline vec2 operator - (const vec2& r) const { return vec2 { x - r.x, y - r.y }; }
	inline vec2 operator * (const vec2& r) const { return vec2 { x * r.x, y * r.y }; }
	inline vec2 operator / (const vec2& r) const { return vec2 { x / r.x, y / r.y }; }
	inline vec2 operator * (const float f) const { return vec2 { x * f, y * f }; }
	inline vec2 operator / (const float f) const { return vec2 { x / f, y / f }; }

	inline bool operator == (const vec2& r) const { return x == r.x && y == r.y; }
};

struct vec3 {
	float x = 0, y = 0, z = 0;

	vec3() = default;
	vec3(float x, float y, float z): x(x), y(y), z(z) {}
	vec3(const vec3&) = default;
	vec3(vec3&&) = default;
	vec3& operator=(const vec3&) = default;
	vec3& operator=(vec3&&) = default;

	inline vec3 operator - () const { return { -x, -y, -z }; }

	inline vec3& operator += (const vec3& o) { *this = *this + o; return *this; }
	inline vec3& operator -= (const vec3& o) { *this = *this - o; return *this; }
	inline vec3& operator *= (const vec3& v) { *this = *this * v; return *this; }
	inline vec3& operator /= (const vec3& v) { *this = *this / v; return *this; }
	inline vec3& operator *= (const float f) { *this = *this * f; return *this; }
	inline vec3& operator /= (const float f) { *this = *this / f; return *this; }

	inline vec3 operator + (const vec3& r) const { return vec3 { x + r.x, y + r.y, z + r.z }; }
	inline vec3 operator - (const vec3& r) const { return vec3 { x - r.x, y - r.y, z - r.z }; }
	inline vec3 operator * (const vec3& r) const { return vec3 { x * r.x, y * r.y, z * r.z }; }
	inline vec3 operator / (const vec3& r) const { return vec3 { x / r.x, y / r.y, z / r.z }; }
	inline vec3 operator * (const float f) const { return vec3 { x * f, y * f, z * f }; }
	inline vec3 operator / (const float f) const { return vec3 { x / f, y / f, z / f }; }

	inline bool operator == (const vec3& r) const { return x == r.x && y == r.y && z == r.z; }

};

struct vec4 {
	float x = 0, y = 0, z = 0, w = 0;

	vec4() = default;
	vec4(float x, float y, float z, float w): x(x), y(y), z(z), w(w) {}
	vec4(const vec4&) = default;
	vec4(vec4&&) = default;
	vec4& operator=(const vec4&) = default;
	vec4& operator=(vec4&&) = default;

	inline vec4 operator - () const { return { -x, -y, -z, -w }; }

	inline vec4& operator += (const vec4& o) { *this = *this + o; return *this; }
	inline vec4& operator -= (const vec4& o) { *this = *this - o; return *this; }
	inline vec4& operator *= (const vec4& v) { *this = *this * v; return *this; }
	inline vec4& operator /= (const vec4& v) { *this = *this / v; return *this; }
	inline vec4& operator *= (const float f) { *this = *this * f; return *this; }
	inline vec4& operator /= (const float f) { *this = *this / f; return *this; }

	inline vec4 operator + (const vec4& r) const { return vec4 { x + r.x, y + r.y, z + r.z, w + r.w }; }
	inline vec4 operator - (const vec4& r) const { return vec4 { x - r.x, y - r.y, z - r.z, w - r.w }; }
	inline vec4 operator * (const vec4& r) const { return vec4 { x * r.x, y * r.y, z * r.z, w * r.w }; }
	inline vec4 operator / (const vec4& r) const { return vec4 { x / r.x, y / r.y, z / r.z, w / r.w }; }
	inline vec4 operator * (const float f) const { return vec4 { x * f, y * f, z * f, w * f }; }
	inline vec4 operator / (const float f) const { return vec4 { x / f, y / f, z / f, w / f }; }

	inline bool operator == (const vec4& r) const { return x == r.x && y == r.y && z == r.z && w == r.w; }
};

struct mat4 {
	float m[16] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	static mat4 identity();
	static mat4 translate(const vec3& pos);
	static mat4 scale(const vec3& scale);
	static mat4 rotate_x(float x);
	static mat4 rotate_y(float y);
	static mat4 rotate_z(float z);
	static mat4 look_at(const vec3& pos, const vec3& at, const vec3& up);
	static mat4 perspective(float angle, float aspect, float zn, float zf);
	static mat4 ortho(float w, float h, float zn, float zf);
	static mat4 world(const vec3& fd, const vec3& up, const vec3& pos);

	mat4 operator * (const mat4& _);
};

float dot(const vec2& u, const vec2& v);
float dot(const vec3& u, const vec3& v);
float dot(const vec4& u, const vec4& v);

vec3 cross(const vec3& u, const vec3& v);

vec2 normalize(const vec2& v);
vec3 normalize(const vec3& v);
vec4 normalize(const vec4& v);

float length(const vec2& v);
float length(const vec3& v);
float length(const vec4& v);

vec2 lerp(const vec2& x, const vec2& y, const float t);
vec3 lerp(const vec3& x, const vec3& y, const float t);
vec4 lerp(const vec4& x, const vec4& y, const float t);

vec2 mul(const vec2& v, const mat4& m);
vec3 mul(const vec3& v, const mat4& m);
vec4 mul(const vec4& v, const mat4& m);
