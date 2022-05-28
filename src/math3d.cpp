#include "math3d.h"
#include <cmath>
#include <cstring>



mat4 mat4::identity() {
	return mat4 {};
}

mat4 mat4::translate(const vec3& pos) {
	return mat4 { {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		pos.x, pos.y, pos.z, 1
	} };
}
mat4 mat4::scale(const vec3& scale) {
	return mat4 { {
		scale.x, 0, 0, 0,
		0, scale.y, 0, 0,
		0, 0, scale.z, 0,
		0, 0, 0, 1
	} };
}
mat4 mat4::rotate_x(float x) {
	float c = cos(x);
	float s = sin(x);
	return mat4 { {
		1, 0, 0, 0,
		0, c, -s, 0,
		0, s, c, 0,
		0, 0, 0, 1
	} };
}
mat4 mat4::rotate_y(float y) {
	float c = cos(y);
	float s = sin(y);
	return mat4 { {
		c, 0, -s, 0,
		0, 1, 0,  0,
		s, 0, c,  0,
		0, 0, 0,  1
	} };
}
mat4 mat4::rotate_z(float z) {
	float c = cos(z);
	float s = sin(z);
	return mat4 {
		c, -s, 0, 0,
		s,  c, 0, 0,
		0,  0, 1, 0,
		0,  0, 0, 1
	};
}
mat4 mat4::look_at(const vec3& pos, const vec3& at, const vec3& up) {
	auto az = normalize(at - pos);
	auto ax = normalize(cross(az, up));
	auto ay = cross(ax, az);

	return mat4 { {
		ax.x,		   ay.x,		  -az.x,	     0,
		ax.y,		   ay.y,		  -az.y,	     0,
		ax.z,		   ay.z,		  -az.z,	     0,
		-dot(ax, pos), -dot(ay, pos), dot(az, pos),  1
	} };
}
mat4 mat4::perspective(float angle, float aspect, float zn, float zf) {
	auto ys = -1.f / tan(angle / 2.f);
	auto xs = ys / aspect;
	auto d = zn - zf;
	auto zs = zf / d;
	auto zb = zn * zf / d;

	return mat4 { {
		xs, 0, 0, 0,
		0, ys, 0, 0,
		0, 0, zs, -1,
		0, 0, zb, 0
	} };
	/*
	* Adapted from GLM source
	
	h = cos(0.5 * rad) / sin(0.5 * rad)
	w = h / aspect;

	[0][0] = w
	[1][1] = h
	[2][2] = zf / (zn - zf)
	[2][3] = -1
	[3][2] = (zf * zn) / (zn - zf);
	*/

	/*
	* 	template<typename T> GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH_ZO(T fov, T width, T height, T zNear, T zFar)
	{
		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zNear - zFar);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}
	*/
}
mat4 mat4::ortho(float w, float h, float zn, float zf) {
	auto zd = zn - zf;
	return mat4 { {
			2 / (w), 0, 0, 0,
			0, 2 / h, 0, 0,
			0, 0, 1 / zd, 0,
			0, 0, zn / zd, 1
	} };
	/*
	Adapted from GLM source
		template<typename T>
		GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_ZO(T left, T right, T bottom, T top, T zNear, T zFar)
		{
			mat<4, 4, T, defaultp> Result(1);
			Result[0][0] = static_cast<T>(2) / (right - left);
			Result[1][1] = static_cast<T>(2) / (top - bottom);
			Result[2][2] = - static_cast<T>(1) / (zFar - zNear);
			Result[3][0] = - (right + left) / (right - left);
			Result[3][1] = - (top + bottom) / (top - bottom);
			Result[3][2] = - zNear / (zFar - zNear);
			return Result;
		}
	*/
}

mat4 mat4::world(const vec3& fd, const vec3& up, const vec3& pos) {
	const auto lt = cross(up, fd);
	return mat4 {{
		lt.x,  up.x,  fd.x,  0,
		lt.y,  up.y,  fd.y,  0,
		lt.z,  up.z,  fd.z,  0,
		pos.x, pos.y, pos.z, 1
	}};
}

mat4 mat4::operator*(const mat4& _) {
	// TODO: vectorize
	return mat4 { {
		m[0] * _.m[0] + m[1] * _.m[4] + m[2] * _.m[8] + m[3] * _.m[12],
		m[0] * _.m[1] + m[1] * _.m[5] + m[2] * _.m[9] + m[3] * _.m[13],
		m[0] * _.m[2] + m[1] * _.m[6] + m[2] * _.m[10] + m[3] * _.m[14],
		m[0] * _.m[3] + m[1] * _.m[7] + m[2] * _.m[11] + m[3] * _.m[15],

		m[4] * _.m[0] + m[5] * _.m[4] + m[6] * _.m[8] + m[7] * _.m[12],
		m[4] * _.m[1] + m[5] * _.m[5] + m[6] * _.m[9] + m[7] * _.m[13],
		m[4] * _.m[2] + m[5] * _.m[6] + m[6] * _.m[10] + m[7] * _.m[14],
		m[4] * _.m[3] + m[5] * _.m[7] + m[6] * _.m[11] + m[7] * _.m[15],

		m[8] * _.m[0] + m[9] * _.m[4] + m[10] * _.m[8] + m[11] * _.m[12],
		m[8] * _.m[1] + m[9] * _.m[5] + m[10] * _.m[9] + m[11] * _.m[13],
		m[8] * _.m[2] + m[9] * _.m[6] + m[10] * _.m[10] + m[11] * _.m[14],
		m[8] * _.m[3] + m[9] * _.m[7] + m[10] * _.m[11] + m[11] * _.m[15],

		m[12] * _.m[0] + m[13] * _.m[4] + m[14] * _.m[8] + m[15] * _.m[12],
		m[12] * _.m[1] + m[13] * _.m[5] + m[14] * _.m[9] + m[15] * _.m[13],
		m[12] * _.m[2] + m[13] * _.m[6] + m[14] * _.m[10] + m[15] * _.m[14],
		m[12] * _.m[3] + m[13] * _.m[7] + m[14] * _.m[11] + m[15] * _.m[15],
	} };
}

float dot(const vec2& u, const vec2& v) {
	return u.x * v.x + u.y * v.y;
}
float dot(const vec3& u, const vec3& v) {
	return u.x * v.x + u.y * v.y + u.z * v.z;
}
float dot(const vec4& u, const vec4& v) {
	return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

vec3 cross(const vec3& u, const vec3& v) {
	return {
		u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x
	};
}

vec2 normalize(const vec2& v) {
	return v / length(v);
}
vec3 normalize(const vec3& v) {
	return v / length(v);
}
vec4 normalize(const vec4& v) {
	return v / length(v);
}

float length(const vec2& v) {
	return sqrt(dot(v, v));
}
float length(const vec3& v) {
	return sqrt(dot(v, v));
}
float length(const vec4& v) {
	return sqrt(dot(v, v));
}


vec2 lerp(const vec2& x, const vec2& y, const float t) {
	return x + (y - x) * t;
}
vec3 lerp(const vec3& x, const vec3& y, const float t) {
	return x + (y - x) * t;
}
vec4 lerp(const vec4& x, const vec4& y, const float t) {
	return x + (y - x) * t;
}
