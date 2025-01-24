#pragma once

#include "cuda_runtime.h"

#include <glm/glm.hpp>

__host__ __device__ __inline__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __inline__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __inline__ float3 operator*(float c, const float3 &a)
{
	return make_float3(c * a.x, c * a.y, c * a.z);
}

__host__ __device__ __inline__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ __inline__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __inline__ float3 operator-(const float3 &a)
{
	return -1.0f * a;
}

__host__ __device__ __inline__ float3 normalize(const float3 &a)
{
	return (1 / sqrt(dot(a, a))) * a;
}

template <typename T> __host__ __device__ __inline__ T max(T a, T b)
{
	return a > b ? a : b;
}
template <typename T> __host__ __device__ __inline__ T min(T a, T b)
{
	return a < b ? a : b;
}

__host__ __device__ __inline__ float3 reflect(const float3 &a, const float3 &n)
{
	return a - 2 * dot(a, n) * n;
}

__host__ __device__ __inline__ float3 operator*(const float3 &a, float c)
{
	return c * a;
}

__host__ __device__ __inline__ float3 operator+=(float3 &a, const float3 &b)
{
	a = a + b;
	return a;
}

__host__ __device__ __inline__ float3 operator-=(float3 &a, const float3 &b)
{
	a = a - b;
	return a;
}

__host__ __device__ __inline__ float3 cross(const float3 &a, const float3 &b)
{
	glm::vec3 a_glm = glm::vec3(a.x, a.y, a.z);
	glm::vec3 b_glm = glm::vec3(b.x, b.y, b.z);
	glm::vec3 result = glm::cross(a_glm, b_glm);
	return make_float3(result.x, result.y, result.z);
}