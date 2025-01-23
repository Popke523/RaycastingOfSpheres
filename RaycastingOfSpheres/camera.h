#pragma once

#include "cuda_runtime.h"

struct camera
{
	float3 position;
	float yaw_degrees;
	float pitch_degrees;
    float fov_degrees;
};

__host__ __device__ __inline__ float3 camera_to_world_rotate(const camera &camera, const float3 &direction)
{
    /**
     * Convert a direction vector from camera space to world space.
     *
     * @param camera The camera.
     * @param direction The direction vector in camera space.
     *
     * @return The direction vector in world space.
     */
    float yaw = camera.yaw_degrees * M_PI / 180.0f;
    float pitch = camera.pitch_degrees * M_PI / 180.0f;

    // Compute the rotation matrix components
    float cy = cosf(yaw);
    float sy = sinf(yaw);
    float cp = cosf(pitch);
    float sp = sinf(pitch);

    // Step 1: Apply pitch rotation
    float3 temp;
    temp.x = direction.x;
    temp.y = direction.y * cp - direction.z * sp;
    temp.z = direction.y * sp + direction.z * cp;

    // Step 2: Apply yaw rotation
    float3 direction_world;
    direction_world.x = temp.x * cy + temp.z * sy;
    direction_world.y = temp.y;
    direction_world.z = -temp.x * sy + temp.z * cy;

    return direction_world;
}
