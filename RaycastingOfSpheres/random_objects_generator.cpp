#include "random_objects_generator.h"

sphere random_sphere(float min_pos, float max_pos, float min_radius, float max_radius)
{
	float3 position = random_float3(min_pos, max_pos);
	float r = random_float(min_radius, max_radius);
	float3 color = random_float3(0.0f, 1.0f);
	float kd = random_float(0.1f, 0.5f);
	float ks = random_float(0.0f, 1.0f);
	float ka = 0.01f;
	float alpha = random_float(2.0f, 40.0f);
	return { position, r, color, kd, ks, ka, alpha };
}

lightSource random_light_source(float min_pos, float max_pos)
{
	float3 pos = random_float3(min_pos, max_pos);
	float3 color = random_float3(0.0f, 1.0f);
	float intensity = 2.0f;
	return { pos, color, intensity };
}

float3 random_float3(float min, float max)
{
	return make_float3(random_float(min, max), random_float(min, max), random_float(min, max));
}

float random_float(float min, float max)
{
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min) + static_cast<float>(min);
}