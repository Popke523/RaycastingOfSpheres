#pragma once

#include <cmath>
#include "cuda_runtime.h"
#include "float3Operators.h"

struct sphere
{
	float3 position;
	float radius;
	float3 color;
	float kd;
	float ks;
	float ka;
	float alpha;
};

__host__ __device__ __inline__ bool LineIntersect(const sphere &sphere, const float3 &origin, const float3 &direction_unit_vector, float &d1, float &d2)
{
	/**
	 * Calculate points of intersection of the sphere with the line given by origin and direction_unit_vector.
	 *
	 * The points of intersection are origin + d1 * direction_unit_vector and origin + d2 * direction_unit_vector.
	 *
	 * @param origin The origin of the line.
	 * @param direction_unit_vector The direction of the line.
	 * @param d1 The parameter of the first point of intersection.
	 * @param d2 The parameter of the second point of intersection.
	 *
	 * @return True if the line intersects the sphere, false otherwise.
	 */
	float3 oc = origin - sphere.position;
	float b = dot(oc, direction_unit_vector);
	float c = dot(oc, oc) - sphere.radius * sphere.radius;
	float disc = b * b - c;
	if (disc < 0) return false;
	disc = sqrt(disc);
	d1 = -b - disc;
	d2 = -b + disc;
	return true;
}