#pragma once

const float pi = 3.1415926535f;
const float degtorad = pi / 180.f;
const float radtodeg = 180.f / pi;

struct vec2 {
	union {
		struct { float x, y; };
		float v[2];
	};

	vec2() { x = 0; y = 0; }
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
	union {
		struct { float x, y, z; };
		float v[3];
	};

	vec3() { x = 0; y = 0; z = 0; }
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
	union {
		struct { float x, y, z, w; };
		float v[4];
	};

	vec4() { x = 0; y = 0; z = 0; w = 0; }
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
	// Right handed look-at (for camera view matrix)
	static mat4 look_at(const vec3& pos, const vec3& at, const vec3& up);
	// Right handed perspective projection with clip space [ 0, 1 ] and inverted Y (for vulkan)
	static mat4 perspective(float angle, float aspect, float zn, float zf);
	static mat4 ortho(float w, float h, float zn, float zf);
	static mat4 world(const vec3& fd, const vec3& up, const vec3& pos);

	mat4 operator * (const mat4& _);
};

struct perlin_noise_1d {};
struct perlin_noise_2d {
	float noise(const vec2& pos);
	float fractal_noise(const vec2& pos);
};
struct perlin_noise_3d {
	float noise(const vec3& pos);
	float fractal_noise(const vec3& pos);
};
struct perlin_noise_4d {};
struct simplex_noise_2d {
	float noise();
	float fractal_noise();
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

float distance(const vec2& a, const vec2& b);
float distance(const vec3& a, const vec3& b);
float distance(const vec4& a, const vec4& b);

vec2 lerp(const vec2& x, const vec2& y, const float t);
vec3 lerp(const vec3& x, const vec3& y, const float t);
vec4 lerp(const vec4& x, const vec4& y, const float t);

vec2 mul(const vec2& v, const mat4& m);
vec3 mul(const vec3& v, const mat4& m);
vec4 mul(const vec4& v, const mat4& m);
vec3 mul_norm(const vec3& v, const mat4& m);

mat4 transpose(const mat4& m);
mat4 inverse(const mat4& m);
float determinant(const mat4& m);

vec3 reflect(const vec3& v, const vec3& n);
vec3 refract(const vec3& v, const vec3& n, float i);



// call a function per component, return value per component
template<typename U> vec2 per_comp(const vec2& v, U u) {
	return vec2 { u(v.x), u(v.y) };
}
template<typename U> vec3 per_comp(const vec3& v, U u) {
	return vec3 { u(v.x), u(v.y), u(v.z) };
}
template<typename U> vec4 per_comp(const vec4& v, U u) {
	return vec4 { u(v.x), u(v.y), u(v.z), u(v.w) };
}

inline float degrees(float r) {
	return r * radtodeg;
}
inline float radians(float d) {
	return d * degtorad;
}
//
//#define decl_per_comp(func) \
//	vec2 func(const vec2& t);\
//	vec3 func(const vec3& t);\
//	vec4 func(const vec4& t);
//#define impl_per_comp(func) \
//	vec2 func(const vec2& t) { return per_comp<vec3, decltype(func)>(t, u); }\
//	vec3 func(const vec3& t) { return per_comp<vec3, decltype(func)>(t, u); }\
//	vec4 func(const vec4& t) { return per_comp<vec4, decltype(func)>(t, u); }
//
//decl_per_comp(abs);
//decl_per_comp(ceil);
//decl_per_comp(atan);
//decl_per_comp(asin);
//decl_per_comp(acos);
//decl_per_comp(atan);
//// decl_per_comp(atan2);
//decl_per_comp(sin);
//decl_per_comp(cos);
//decl_per_comp(tan);
//
//decl_per_comp(degrees);
//decl_per_comp(radians);
//
//decl_per_comp(exp);
//decl_per_comp(exp2);
//decl_per_comp(floor);
//decl_per_comp(log);
//decl_per_comp(log2);
//decl_per_comp(log10);
//decl_per_comp(pow);
//decl_per_comp(round);
//decl_per_comp(saturate);
//decl_per_comp(sign);
//decl_per_comp(smoothstep);
//decl_per_comp(smootherstep);
//decl_per_comp(sqrt);

// all
// any
// clamp
// atan2
// min
// max
// fmod
// frac
// frexp
// modf
// slerp

/*

matrix
	frustum (??)
	perspective (infinite (??), lh, rh, fov, ...), ortho (lh, rh)
	look_at
	pick_matrix (??)
	project (??) unproject (??)
	rotate_x, rotate_y, rotate_z, rotate_axis, rotate_euler
	scale
	translate

color stuff
	linear to srgb etc

packing stuff
	byte
	byten
	ubyten
	short
	shortn
	ushortn
	half

some intersection stuff?
	closest_point(line, point)
	line_sphere
	line_box
	line_triangle
	line_plane

*/